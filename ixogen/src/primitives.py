from dataclasses import dataclass
from typing import List

import hashlib

from ixogen import consensus, script, tx


@dataclass
class BlockHeader:
    """A Ixogen block header."""

    version: int
    hashPrevBlock: bytes
    hashMerkleRoot: bytes
    nTime: int
    nBits: int
    nNonce: int

    def GetHash(self) -> bytes:
        """Returns the hash of the block header."""
        h = hashlib.sha256()
        h.update(self.version.to_bytes(4, byteorder="big"))
        h.update(self.hashPrevBlock)
        h.update(self.hashMerkleRoot)
        h.update(self.nTime.to_bytes(4, byteorder="big"))
        h.update(self.nBits.to_bytes(4, byteorder="big"))
        h.update(self.nNonce.to_bytes(4, byteorder="big"))
        return h.digest()


@dataclass
class Block(BlockHeader):
    """A Ixogen block."""

    vtx: List[tx.Tx]

    def ToString(self) -> str:
        """Returns a string representation of the block."""
        s = f"Block(hash={self.GetHash().hex()}, ver={self.version}, hashPrevBlock={self.hashPrevBlock.hex()}, hashMerkleRoot={self.hashMerkleRoot.hex()}, " \
            f"nTime={self.nTime}, nBits={self.nBits:08x}, nNonce={self.nNonce}, vtx={len(self.vtx)})\n"
        for tx in self.vtx:
            s += f"  {tx.ToString()}\n"
        return s


@dataclass
class BlockLocator:
    """A block locator describes a place in the block chain to another node."""

    vHave: List[bytes]

    def SetNull(self):
        self.vHave = []

    def IsNull(self) -> bool:
        return len(self.vHave) == 0

#---------------------------------------------------------------------------------------------------------------------------
#TRANSACTION
from dataclasses import dataclass
from typing import List

from ixogen import consensus, script, tx


@dataclass
class OutPoint:
    """A Ixogen outpoint."""

    hash: bytes
    n: int

    def ToString(self) -> str:
        """Returns a string representation of the outpoint."""
        return f"COutPoint({self.hash.hex()[:10]}, {self.n})"


@dataclass
class TxIn:
    """A Ixogen transaction input."""

    prevout: OutPoint
    scriptSig: script.Script
    nSequence: int

    def ToString(self) -> str:
        """Returns a string representation of the transaction input."""
        s = f"CTxIn({self.prevout.ToString()}"
        if self.prevout.IsNull():
            s += f", coinbase {self.scriptSig.hex()}"
        else:
            s += f", scriptSig={self.scriptSig.hex()[:24]}"
        if self.nSequence != consensus.SEQUENCE_FINAL:
            s += f", nSequence={self.nSequence}"
        s += ")"
        return s


@dataclass
class TxOut:
    """A Ixogen transaction output."""

    nValue: int
    scriptPubKey: script.Script

    def ToString(self) -> str:
        """Returns a string representation of the transaction output."""
        return f"CTxOut(nValue={self.nValue // consensus.COIN}.{self.nValue % consensus.COIN}, scriptPubKey={self.scriptPubKey.hex()[:30]})"


@dataclass
class MutableTransaction:
    """A mutable Ixogen transaction."""

    nVersion: int
    vin: List[TxIn]
    vout: List[TxOut]
    nLockTime: int

    def GetHash(self) -> bytes:
        """Returns the hash of the transaction."""
        return (tx.CHashWriter(serializer.SerializationType.TRANSACTION_NO_WITNESS) << self).GetHash()


@dataclass
class Transaction(MutableTransaction):
    """A Ixogen transaction."""

    hash: bytes
    m_witness_hash: bytes

    def ComputeHash(self) -> bytes:
        """Returns the hash of the transaction."""
        return (tx.CHashWriter(serializer.SerializationType.TRANSACTION_NO_WITNESS) << self).GetHash()

    def ComputeWitnessHash(self) -> bytes:
        """Returns the witness hash of the transaction."""
        if not self.HasWitness():
            return self.hash
        return (tx.CHashWriter(serializer.SerializationType.Witness) << self).GetHash()

    def GetValueOut(self) -> int:
        """Returns the total value of the transaction outputs."""
        nValueOut = 0
        for tx_out in self.vout:
            if not consensus.MoneyRange(tx_out.nValue) or not consensus.MoneyRange(nValueOut + tx_out.nValue):
                raise ValueError("value out of range")
            nValueOut += tx_out.nValue
        return nValueOut

    def GetTotalSize(self) -> int:
        """Returns the total size of the transaction in bytes."""
        return tx.GetSerializeSize(self)

    def ToString(self) -> str:
        """Returns a string representation of the transaction."""
        s = f"CTransaction(hash={self.hash.hex()[:10]}, ver={self.nVersion}, vin.size={len(self.vin)}, vout.size={len(self.vout)}, nLockTime={self.nLockTime})\n"
        for tx_in in self.vin:
            s += f"    {tx_in.ToString()}\n"
        for tx_in in self.vin:
            s += f"    {tx_in.scriptWitness.ToString()}\n"
        for tx_out in self.vout:
            s += f"    {tx_out.ToString()}\n"
        return s
#------------------------------------------------------------------------------------------
@dataclass
class TxIn:
    """A Ixogen transaction input."""

    prevout: OutPoint
    scriptSig: script.Script
    nSequence: int

    def ToString(self) -> str:
        """Returns a string representation of the transaction input."""
        s = f"CTxIn({self.prevout.ToString()}"
        if self.prevout.IsNull():
            s += f", coinbase {self.scriptSig.hex()}"
        else:
            s += f", scriptSig={self.scriptSig.hex()[:24]}"
        if self.nSequence != consensus.SEQUENCE_FINAL:
            s += f", nSequence={self.nSequence}"
        s += ")"
        return s


@dataclass
class TxOut:
    """A Ixogen transaction output."""

    nValue: int
    scriptPubKey: script.Script

    def ToString(self) -> str:
        """Returns a string representation of the transaction output."""
        return f"CTxOut(nValue={self.nValue // consensus.COIN}.{self.nValue % consensus.COIN}, scriptPubKey={self.scriptPubKey.hex()[:30]})"


@dataclass
class Transaction:
    """A Ixogen transaction."""

    vin: List[TxIn]
    vout: List[TxOut]
    nVersion: int
    nLockTime: int
    hash: bytes
    m_witness_hash: bytes

    def ComputeHash(self) -> bytes:
        """Returns the hash of the transaction."""
        return (tx.CHashWriter(serializer.SerializationType.TRANSACTION_NO_WITNESS) << self).GetHash()

    def ComputeWitnessHash(self) -> bytes:
        """Returns the witness hash of the transaction."""
        if not self.HasWitness():
            return self.hash
        return (tx.CHashWriter(serializer.SerializationType.Witness) << self).GetHash()

    def __eq__(self, other: Transaction) -> bool:
        return self.hash == other.hash

    def ToString(self) -> str:
        """Returns a string representation of the transaction."""
        s = f"CTransaction(hash={self.hash.hex()[:10]}, ver={self.nVersion}, vin.size={len(self.vin)}, vout.size={len(self.vout)}, nLockTime={self.nLockTime})\n"
        for tx_in in self.vin:
            s += f"    {tx_in.ToString()}\n"
        for tx_out in self.vout:
            s += f"    {tx_out.ToString()}\n"
        return s


def CalculateOutputValue(tx: Transaction) -> int:
    """Calculates the total value of the transaction outputs."""
    return sum(tx_out.nValue for tx_out in tx.vout)

@dataclass
class OutPoint:
    """A Ixogen outpoint."""

    hash: bytes
    n: int

    def ToString(self) -> str:
        """Returns a string representation of the outpoint."""
        return f"COutPoint({self.hash.hex()[:10]}, {self.n})"


@dataclass
class TxIn:
    """A Ixogen transaction input."""

    prevout: OutPoint
    scriptSig: script.Script
    nSequence: int

    def ToString(self) -> str:
        """Returns a string representation of the transaction input."""
        s = f"CTxIn({self.prevout.ToString()}"
        if self.prevout.IsNull():
            s += f", coinbase {self.scriptSig.hex()}"
        else:
            s += f", scriptSig={self.scriptSig.hex()[:24]}"
        if self.nSequence != consensus.SEQUENCE_FINAL:
            s += f", nSequence={self.nSequence}"
        s += ")"
        return s


@dataclass
class TxOut:
    """A Ixogen transaction output."""

    nValue: int
    scriptPubKey: script.Script

    def ToString(self) -> str:
        """Returns a string representation of the transaction output."""
        return f"CTxOut(nValue={self.nValue // consensus.COIN}.{self.nValue % consensus.COIN}, scriptPubKey={self.scriptPubKey.hex()[:30]})"


@dataclass
class CMutableTransaction:
    """A mutable Ixogen transaction."""

    vin: List[TxIn]
    vout: List[TxOut]
    nVersion: int
    nLockTime: int

    def GetHash(self) -> bytes:
        """Returns the hash of the transaction."""
        return (tx.CHashWriter(serializer.SerializationType.TRANSACTION_NO_WITNESS) << self).GetHash()

    def HasWitness(self) -> bool:
        """Returns whether the transaction has witness data."""
        for tx_in in self.vin:
            if not tx_in.scriptWitness.IsNull():
                return True
        return False


@dataclass
class CTransaction(CMutableTransaction):
    """A Ixogen transaction."""

    hash: bytes
    m_witness_hash: bytes

    def __eq__(self, other: CTransaction) -> bool:
        return self.hash == other.hash

    def ToString(self) -> str:
        """Returns a string representation of the transaction."""
        s = f"CTransaction(hash={self.hash.hex()[:10]}, ver={self.nVersion}, vin.size={len(self.vin)}, vout.size={len(self.vout)}, nLockTime={self.nLockTime})\n"
        for tx_in in self.vin:
            s += f"    {tx_in.ToString()}\n"
        for tx_out in self.vout:
            s += f"    {tx_out.ToString()}\n"
        return s

