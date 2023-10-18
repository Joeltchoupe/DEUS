import leveldb

class Block:
    def __init__(self, contents):
        self.data = contents.data
        self.size = contents.size
        self.owned = contents.heap_allocated

        if self.size < 4:
            self.size = 0  # Error marker
        else:
            max_restarts_allowed = (self.size - 4) // 4
            if self.NumRestarts() > max_restarts_allowed:
                # The size is too small for NumRestarts()
                self.size = 0
            else:
                self.restart_offset = self.size - (1 + self.NumRestarts()) * 4

    def __del__(self):
        if self.owned:
            del self.data

    def NumRestarts(self):
        assert self.size >= 4
        return leveldb.DecodeFixed32(self.data + self.size - 4)

    def NewIterator(self, comparator):
        if self.size < 4:
            return leveldb.NewErrorIterator(leveldb.Status.Corruption("bad block contents"))
        num_restarts = self.NumRestarts()
        if num_restarts == 0:
            return leveldb.NewEmptyIterator()
        else:
            return leveldb.Iter(comparator, self.data, self.restart_offset, num_restarts)



    def ParseEntry(self, p, limit):
        """Decode the next entry starting at "p",
        storing the number of shared key bytes, non_shared key bytes,
        and the length of the value in "*shared", "*non_shared", and
        "*value_length", respectively.  Will not dereference past "limit".

        If any errors are detected, returns nullptr.  Otherwise, returns a
        pointer to the key delta (just past the three decoded values).
        """
        if limit - p < 3:
            return nullptr
        shared = reinterpret_cast<const uint8_t*>(p)[0]
        non_shared = reinterpret_cast<const uint8_t*>(p)[1]
        value_length = reinterpret_cast<const uint8_t*>(p)[2]
        if ((*shared | *non_shared | *value_length) < 128):
            # Fast path: all three values are encoded in one byte each
            p += 3
        else:
            if ((p = leveldb.GetVarint32Ptr(p, limit, shared)) == nullptr):
                return nullptr
            if ((p = leveldb.GetVarint32Ptr(p, limit, non_shared)) == nullptr):
                return nullptr
            if ((p = leveldb.GetVarint32Ptr(p, limit, value_length)) == nullptr):
                return nullptr

        if (static_cast<uint32_t>(limit - p) < (*non_shared + *value_length)):
            return nullptr
        return p

    def Iter(self, comparator):
        """Returns an iterator over the block.

        The returned iterator is positioned at the first entry in the block.
        The iterator is invalidated if the block is modified.
        """
        return BlockIter(comparator, self.data, self.restart_offset,
                         self.NumRestarts())

    class BlockIter(leveldb.Iterator):
        """An iterator over a block.

        The iterator is positioned at the first entry in the block when
        first created.  It is invalidated if the block is modified.
        """

        def __init__(self, comparator, data, restart_offset, num_restarts):
            super().__init__(comparator)
            self.data = data
            self.restart_offset = restart_offset
            self.num_restarts = num_restarts
            self.restart_index = 0
            self.current_offset = restart_offset
            self.key = b""
            self.value = b""

        def Valid(self) -> bool:
            return self.current_offset < self.restart_offset

        def key(self) -> bytes:
            return self.key

        def value(self) -> bytes:
            return self.value

        def Next(self):
            assert self.Valid()

            while True:
                if self.current_offset >= self.restart_offset:
                    return False

                p = self.data + self.current_offset
                const char* limit = self.data + self.restart_offset;  // Restarts come right after data
                if p >= limit:
                    // No more entries to return.  Mark as invalid.
                    self.current_offset = self.restart_offset
                    self.restart_index = self.num_restarts
                    self.key = b""
                    self.value = b""
                    return False

                // Decode next entry
                uint32_t shared, non_shared, value_length;
                p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
                if p == nullptr || self.key.size() < shared:
                    self.CorruptionError()
                    return False
                else:
                    self.key = self.key[:shared] + p[:non_shared]
                    self.value = p + non_shared
                    self.current_offset = NextEntryOffset()
                    return True

        def Prev(self):
            assert self.Valid()

            while True:
                if self.restart_index == 0:
                    // No more entries
                    self.current_offset = self.restart_offset
                    self.restart_index = self.num_restarts
                    self.key = b""
                    self.value = b""
                    return False

                self.restart_index -= 1
                self.current_offset = GetRestartPoint(self.restart_index)
                while True:
                    if self.current_offset >= self.restart_offset:
                        return False

                    p = self.data + self.current_offset
                    const char* limit = self.data + self.restart_offset;  // Restarts come right after data
                    if p >= limit:
                        // No more entries to return.  Mark as invalid.
                        self.current_offset = self.restart_offset
                        self.restart_index = self.num_restarts
                        self.key = b""
                        self.value = b""
                        return False

                    // Decode next entry
                    uint32_t shared, non_shared, value_length;
                    p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
                    if p == nullptr || self.key.size() < shared:
                        self.CorruptionError()
                        return False
                    else:
                        self.key = self.key[:shared] + p[:non_shared]
                        self.value = p + non_shared
                        self.current_offset = NextEntryOffset()
                        return True

       def Seek(self, target):
            # Binary search in restart array to find the last restart point
            # with a key < target
            uint32_t left = 0;
            uint32_t right = self.num_restarts - 1;
            while left < right:
                uint32_t mid = (left + right + 1) // 2;
                uint32_t region_offset = GetRestartPoint(mid)
                uint32_t shared, non_shared, value_length;
                p = DecodeEntry(self.data + region_offset, self.data + self.restart_offset, &shared, &non_shared, &value_length);
                if p == nullptr || self.key.size() < shared:
                    self.CorruptionError()
                    return False
                else:
                    if comparator.Compare(key, p[:shared] + p[shared:shared + non_shared]) < 0:
                        right = mid - 1;
                    else:
                        left = mid;

            self.restart_index = left;
            self.current_offset = GetRestartPoint(left);
            self.ParseNextKey();
            return self.Valid();
    def SeekToFirst(self):
            self.SeekToRestartPoint(0)
            self.ParseNextKey()

        def SeekToLast(self):
            self.SeekToRestartPoint(self.num_restarts - 1)
            while self.ParseNextKey():
                # Keep skipping
            self.current_offset = self.restart_offset
            self.restart_index = self.num_restarts
            self.key = b""
            self.value = b""

        def SeekToRestartPoint(self, index):
            self.key = b""
            self.restart_index = index
            # current_ will be fixed by ParseNextKey();

            # ParseNextKey() starts at the end of value_, so set value_ accordingly
            uint32_t offset = GetRestartPoint(index);
            self.value = Slice(self.data + offset, 0)

        def CorruptionError(self):
            self.current_offset = self.restart_offset
            self.restart_index = self.num_restarts
            self.status = leveldb.Status.Corruption("bad entry in block")
            self.key = b""
            self.value = b""

        def ParseNextKey(self):
            current_ = self.NextEntryOffset()
            const char* p = self.data + current_;
            const char* limit = self.data + self.restart_offset;  // Restarts come right after data
            if p >= limit:
                // No more entries to return.  Mark as invalid.
                self.current_offset = self.restart_offset
                self.restart_index = self.num_restarts
                return False

            // Decode next entry
            uint32_t shared, non_shared, value_length;
            p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
            if p == nullptr || self.key.size() < shared:
                self.CorruptionError()
                return False
            else:
                self.key = self.key[:shared] + p[:non_shared]
                self.value = p + non_shared
                self.current_offset = NextEntryOffset()
                return True

        def NextEntryOffset(self):
            return (self.value.data() + self.value.size()) - self.data

        def GetRestartPoint(self, index):
            assert(index < self.num_restarts)
            return DecodeFixed32(self.data + self.restart_offset + index * sizeof(uint32_t))

        def __del__(self):
            if self.owned:
                del self.data


