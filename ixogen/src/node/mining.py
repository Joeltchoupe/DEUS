import json
import hashlib

def get_network_hash_ps(lookup=-1, height=-1, active_chain=None):
    if active_chain is None:
        active_chain = GetActiveChain()

    pb = active_chain.Tip()

    if height >= 0 and height < active_chain.Height():
        pb = active_chain[height]

    if pb is None or not pb.nHeight:
        return 0

    # If lookup is -1, then use blocks since last difficulty change.
    if lookup <= 0:
        lookup = pb.nHeight % Params().GetConsensus().DifficultyAdjustmentInterval() + 1

    # If lookup is larger than chain, then set it to chain length.
    if lookup > pb.nHeight:
        lookup = pb.nHeight

    pb0 = pb
    min_time = pb0.GetBlockTime()
    max_time = min_time
    for i in range(lookup):
        pb0 = pb0.pprev
        time = pb0.GetBlockTime()
        min_time = min(time, min_time)
        max_time = max(time, max_time)

    # In case there's a situation where minTime == maxTime, we don't want a divide by zero exception.
    if min_time == max_time:
        return 0

    work_diff = pb.nChainWork - pb0.nChainWork
    time_diff = max_time - min_time

    return work_diff.getdouble() / time_diff

def generate_blocks(n_generate=1, max_tries=100000000):
    chainman = EnsureAnyChainman()
    mempool = GetMemPool()
    coinbase_script = GetScriptForDestination(scriptPubKey=GetCoinbaseScript())

    block_hashes = []
    while n_generate > 0 and not IsNodeShutdown():
        pblocktemplate = BlockAssembler(chainman.ActiveChainstate(), mempool).CreateNewBlock(coinbase_script)
        if pblocktemplate is None:
            raise JSONRPCError(RPC_INTERNAL_ERROR, "Couldn't create new block")

        block_out = GenerateBlock(chainman, pblocktemplate.block, max_tries, process_new_block=True)
        if block_out is None:
            break

        if block_out:
            n_generate -= 1
            block_hashes.append(block_out.GetHash().GetHex())

    return block_hashes

def get_script_from_descriptor(descriptor):
    error = ""
    script = CScript()

    success = getScriptFromDescriptor(descriptor, script, error)
    if not success:
        raise JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, error)

    return script
def generatetodescriptor(num_blocks, descriptor, maxtries=DEFAULT_MAX_TRIES):

    coinbase_script = get_script_from_descriptor(descriptor)
    chainman = EnsureChainman()
    mempool = EnsureMemPool()

    return generateBlocks(chainman, mempool, coinbase_script, num_blocks, maxtries)

def generatetoaddress(num_blocks, address, maxtries=DEFAULT_MAX_TRIES):

    destination = DecodeDestination(address)
    if not IsValidDestination(destination):
        raise JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Error: Invalid address")

    coinbase_script = GetScriptForDestination(destination)
    chainman = EnsureChainman()
    mempool = EnsureMemPool()

    return generateBlocks(chainman, mempool, coinbase_script, num_blocks, maxtries)

def generateblock(address_or_descriptor, raw_txs_or_txids, submit=True):

    coinbase_script = get_script_from_descriptor(address_or_descriptor)
    node = EnsureAnyNodeContext()
    mempool = EnsureMemPool(node)

    txs = []
    for str in raw_txs_or_txids:

        hash = hashlib.sha256(str.encode('utf-8')).hexdigest()
        tx = mempool.get(hash)
        if tx is None:
            raise JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, f"Transaction {str} not in mempool.")

        txs.append(tx)

    chainman = EnsureChainman(node)

    with LOCK(cs_main):
        blocktemplate = BlockAssembler(chainman.ActiveChainstate(), None).CreateNewBlock(coinbase_script)
        if blocktemplate is None:
            raise JSONRPCError(RPC_INTERNAL_ERROR, "Couldn't create new block")

        block = blocktemplate.block

    block.vtx.insert(block.vtx.end(), txs)
    RegenerateCommitments(block, chainman)

    with LOCK(cs_main):
        state = BlockValidationState()
        if not TestBlockValidity(state, chainman.GetParams(), chainman.ActiveChainstate(), block, chainman.m_blockman.LookupBlockIndex(block.hashPrevBlock), GetAdjustedTime, False, False):
            raise JSONRPCError(RPC_VERIFY_ERROR, f"TestBlockValidity failed: {state.ToString()}")

    block_out = None
    max_tries = DEFAULT_MAX_TRIES

    if GenerateBlock(chainman, block, max_tries, block_out, submit) and block_out is not None:

        obj = {}
        obj["hash"] = block_out.GetHash().GetHex()

        if not submit:
            block_ser = CDataStream(SER_NETWORK, PROTOCOL_VERSION | RPCSerializationFlags())
            block_ser << block_out
            obj["hex"] = HexStr(block_ser)

        return obj

    else:
        raise JSONRPCError(RPC_MISC_ERROR, "Failed to make block.")

  def getmininginfo():

    node = EnsureAnyNodeContext()
    mempool = EnsureMemPool(node)
    chainman = EnsureChainman(node)

    with LOCK(cs_main):
        active_chain = chainman.ActiveChain()

        obj = {}
        obj["blocks"] = active_chain.Height()
        if BlockAssembler.m_last_block_weight is not None:
            obj["currentblockweight"] = BlockAssembler.m_last_block_weight

        if BlockAssembler.m_last_block_num_txs is not None:
            obj["currentblocktx"] = BlockAssembler.m_last_block_num_txs

        obj["difficulty"] = GetDifficulty(active_chain.Tip())
        obj["networkhashps"] = getnetworkhashps().HandleRequest()
        obj["pooledtx"] = len(mempool)
        obj["chain"] = chainman.GetParams().GetChainTypeString()
        obj["warnings"] = GetWarnings(False).original

        return obj
#
def prioritisetransaction(txid, fee_delta):

    with LOCK(cs_main):

        EnsureAnyMemPool().PrioritiseTransaction(txid, fee_delta)

        return True

def getprioritisedtransactions():

    node = EnsureAnyNodeContext()
    mempool = EnsureMemPool(node)

    rpc_result = {}
    for delta_info in mempool.GetPrioritisedTransactions():

        result_inner = {}
        result_inner["fee_delta"] = delta_info.delta
        result_inner["in_mempool"] = delta_info.in_mempool

        rpc_result[delta_info.txid.GetHex()] = result_inner

    return rpc_result

def BIP22ValidationResult(state):

    if state.IsValid():
        return None

    if state.IsError():
        raise JSONRPCError(RPC_VERIFY_ERROR, state.ToString())

    if state.IsInvalid():

        strRejectReason = state.GetRejectReason()
        if strRejectReason.empty():
            return "rejected"

        return strRejectReason

    # Should be impossible
    return "valid?"

def gbt_vb_name(pos):

    vbinfo = VersionBitsDeploymentInfo[pos]
    s = vbinfo.name

    if not vbinfo.gbt_force:
        s = "!" + s

    return s

def getblocktemplate(template_request):

    if "capabilities" in template_request and "proposal" in template_request["capabilities"]:
        raise NotImplementedError("Proposal mode is not supported")

    node = EnsureAnyNodeContext()
    chainman = EnsureChainman(node)

    with LOCK(cs_main):

        blocktemplate = BlockAssembler(chainman.ActiveChainstate(), nullptr).CreateNewBlock(template_request)
        if blocktemplate is None:
            raise JSONRPCError(RPC_INTERNAL_ERROR, "Couldn't create new block")

        obj = {}
        obj["version"] = blocktemplate.block.nVersion
        obj["rules"] = blocktemplate.block.vchBlockVersion.ToString()
        obj["vbavailable"] = blocktemplate.vbavailable
        obj["capabilities"] = blocktemplate.vchBlockVersion.ToString()
        obj["vbrequired"] = blocktemplate.vbrequired
        obj["previousblockhash"] = blocktemplate.block.hashPrevBlock.GetHex()
        obj["transactions"] = []

        for tx in blocktemplate.block.vtx:
            obj["transactions"].append({
                "data": tx.serialize().hex(),
                "txid": tx.GetHash().GetHex(),
                "hash": tx.GetWitnessHash().GetHex(),
                "depends": tx.vin,
                "fee": tx.GetFee(),
                "sigops": tx.GetSigOpCount(),
                "weight": tx.GetWeight()
            })

        obj["coinbaseaux"] = blocktemplate.coinbaseaux
        obj["coinbasevalue"] = blocktemplate.block.vtx[0].vout[0].nValue
        obj["longpollid"] = blocktemplate.longpollid
        obj["target"] = blocktemplate.block.hashMerkleRoot.GetHex()
        obj["mintime"] = blocktemplate.nTime
        obj["mutable"] = blocktemplate.vchBlockVersion.ToString()
        obj["noncerange"] = blocktemplate.nNonceRange
        obj["sigoplimit"] = blocktemplate.nSigOpLimit
        obj["sizelimit"] = blocktemplate.nSizeLimit
        obj["weightlimit"] = blocktemplate.nWeightLimit
        obj["curtime"] = blocktemplate.nTime
        obj["bits"] = blocktemplate.block.nBits
        obj["height"] = blocktemplate.nHeight

        if node.signet_challenge is not None:
            obj["signet_challenge"] = node.signet_challenge.GetHex()

        if blocktemplate.default_witness_commitment is not None:
            obj["default_witness_commitment"] = blocktemplate.default_witness_commitment.GetHex()

        return obj

  def getblocktemplate(node, request):

    chainman = EnsureChainman(node)
    LOCK(cs_main)

    strMode = "template"
    lpval = None
    setClientRules = set()
    active_chainstate = chainman.ActiveChainstate()
    active_chain = active_chainstate.m_chain
    if request.params[0] is not None:
        oparam = request.params[0].get_obj()
        modeval = oparam.find_value("mode")
        if modeval is not None and modeval.isStr():
            strMode = modeval.get_str()
        elif modeval is None:
            pass
        else:
            raise JSONRPCError(RPC_INVALID_PARAMETER, "Invalid mode")
        lpval = oparam.find_value("longpollid")

        if strMode == "proposal":
            dataval = oparam.find_value("data")
            if dataval is None or not dataval.isStr():
                raise JSONRPCError(RPC_TYPE_ERROR, "Missing data String key for proposal")

            block = CBlock()
            if not DecodeHexBlk(block, dataval.get_str()):
                raise JSONRPCError(RPC_DESERIALIZATION_ERROR, "Block decode failed")

            hash = block.GetHash()
            pindex = chainman.m_blockman.LookupBlockIndex(hash)
            if pindex:
                if pindex.IsValid(BLOCK_VALID_SCRIPTS):
                    return "duplicate"
                if pindex.nStatus & BLOCK_FAILED_MASK:
                    return "duplicate-invalid"
                return "duplicate-inconclusive"

            pindexPrev = active_chain.Tip()
            # TestBlockValidity only supports blocks built on the current Tip
            if block.hashPrevBlock != pindexPrev.GetBlockHash():
                return "inconclusive-not-best-prevblk"
            state = BlockValidationState()
            TestBlockValidity(state, chainman.GetParams(), active_chainstate, block, pindexPrev, GetAdjustedTime, False, True)
            return BIP22ValidationResult(state)

        aClientRules = oparam.find_value("rules")
        if aClientRules is not None and aClientRules.isArray():
            for v in aClientRules:
                setClientRules.add(v.get_str())

    if strMode != "template":
        raise JSONRPCError(RPC_INVALID_PARAMETER, "Invalid mode")

    if not chainman.GetParams().IsTestChain():
        connman = EnsureConnman(node)
        if connman.GetNodeCount(ConnectionDirection.Both) == 0:
            raise JSONRPCError(RPC_CLIENT_NOT_CONNECTED, PACKAGE_NAME + " is not connected!")

        if chainman.IsInitialBlockDownload():
            raise JSONRPCError(RPC_CLIENT_IN_INITIAL_DOWNLOAD, PACKAGE_NAME + " is in initial sync and waiting for blocks...")

    static unsigned int nTransactionsUpdatedLast
    mempool = EnsureMemPool(node)

    if lpval is not None:
        # Wait to respond until either the best block changes, OR a minute has passed and there are more transactions
        hashWatchedChain = None
        checktxtime = None
        unsigned int nTransactionsUpdatedLastLP

        if lpval.isStr():
            # Format: <hashBestChain><nTransactionsUpdatedLast>
            lpstr = lpval.get_str()

            hashWatchedChain = ParseHashV(lpstr[:64], "longpollid")
            nTransactionsUpdatedLastLP = LocaleIndependentAtoi(lpstr[64:])
        else:
            # NOTE: Spec does not specify behaviour for non-string longpollid, but this makes testing easier
            hashWatchedChain = active_chain.Tip().GetBlockHash()
            nTransactionsUpdatedLastLP = nTransactionsUpdatedLast

        # Release lock while waiting
        LEAVE_CRITICAL_SECTION(cs_main)
        {
            checktxtime = datetime.datetime.now() + datetime.timedelta(minutes=1)

            WAIT_LOCK(g_best_block_mutex, lock)
            while g_best_block == hashWatchedChain and IsRPCRunning():
                if g_best_block_cv.wait_until(lock, checktxtime) == std.cv_status::timeout:
                    #    # Timeout: Check transactions for update
    # without holding the mempool lock to avoid deadlocks
    if mempool.GetTransactionsUpdated() != nTransactionsUpdatedLastLP:
        break
    checktxtime += datetime.timedelta(seconds=10)
        }
        ENTER_CRITICAL_SECTION(cs_main)

        if not IsRPCRunning():
            raise JSONRPCError(RPC_CLIENT_NOT_CONNECTED, "Shutting down")
        # TODO: Maybe recheck connections/IBD and (if something wrong) send an expires-immediately template to stop miners?

    const Consensus::Params& consensusParams = chainman.GetParams().GetConsensus();

    # GBT must be called with 'signet' set in the rules for signet chains
    if consensusParams.signet_blocks and setClientRules.count("signet") != 1:
        raise JSONRPCError(RPC_INVALID_PARAMETER, "getblocktemplate must be called with the signet rule set (call with {\"rules\": [\"segwit\", \"signet\"]})");

    # GBT must be called with 'segwit' set in the rules
    if setClientRules.count("segwit") != 1:
        raise JSONRPCError(RPC_INVALID_PARAMETER, "getblocktemplate must be called with the segwit rule set (call with {\"rules\": [\"segwit\"]})");

    # Update block
    static CBlockIndex* pindexPrev
    static int64_t time_start
    static std::unique_ptr<CBlockTemplate> pblocktemplate
    if pindexPrev != active_chain.Tip() or
        (mempool.GetTransactionsUpdated() != nTransactionsUpdatedLast and GetTime() - time_start > 5):
        # Clear pindexPrev so future calls make a new block, despite any failures from here on
        pindexPrev = nullptr

        # Store the pindexBest used before CreateNewBlock, to avoid races
        nTransactionsUpdatedLast = mempool.GetTransactionsUpdated()
        CBlockIndex* pindexPrevNew = active_chain.Tip()
        time_start = GetTime()

        # Create new block
        CScript scriptDummy = CScript() << OP_TRUE;
        pblocktemplate = BlockAssembler{active_chainstate, &mempool}.CreateNewBlock(scriptDummy);
        if pblocktemplate is None:
            raise JSONRPCError(RPC_OUT_OF_MEMORY, "Out of memory");

        # Need to update only after we know CreateNewBlock succeeded
        pindexPrev = pindexPrevNew
    CHECK_NONFATAL(pindexPrev)
    CBlock* pblock = &pblocktemplate->block; // pointer for convenience

    # Update nTime
    UpdateTime(pblock, consensusParams, pindexPrev);
    pblock->nNonce = 0;

    # Create coinbase transaction
    CMutableTransaction coinbaseTx;
    coinbaseTx.vin.resize(1);
    coinbaseTx.vin[0].prevout.SetNull();
    coinbaseTx.vin[0].scriptSig = CScript() << OP_0 << std::to_string(pblock->nHeight);
    coinbaseTx.vout.resize(1);
    coinbaseTx.vout[0].scriptPubKey = CScript() << OP_TRUE;
    coinbaseTx.vout[0].nValue = blockreward(consensusParams, pindexPrev->nHeight);
    pblock->vtx.push_back(coinbaseTx);

    # Fill in header
    pblock->nVersion = consensusParams.nBlockVersion;
    pblock->nBits = consensusParams.nPowLimit;
    pblock->nTime = GetAdjustedTime(pindexPrev);
    pblock->nNonce = 0;

    # Sign block
    std::vector<unsigned char> signature;
    if (!SignBlock(*pblock, consensusParams, signature))
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Unable to sign block");

    # Build return object
        # Build return object
    obj = {
        "version": pblock->nVersion,
        "hash": pblock->GetHash().hex(),
        "previousblockhash": pblock->hashPrevBlock.hex(),
        "merkleroot": pblock->hashMerkleRoot.hex(),
        "time": pblock->nTime,
        "bits": pblock->nBits,
        "nonce": pblock->nNonce,
        "coinbasetxn": pblock->vtx[0].hex(),
        "longpollid": GetRandHash().hex(),
        "target": pblock->GetTarget().hex(),
        "mintime": pblock->nTime,
        "mutable": consensusParams.nBlockMaxSigOps,
        "noncerange": pblock->nNonceRange,
        "sigoplimit": consensusParams.nMaxBlockSigOps,
        "sizelimit": consensusParams.nMaxBlocksize,
        "weightlimit": consensusParams.nMaxBlockWeight,
        "curtime": GetTime(),
        "height": pblock->nHeight,
    }

    # Signet specific fields
    if consensusParams.signet_blocks:
        obj["signet_challenge"] = consensusParams.signet_challenge.hex()
        if pblocktemplate.signet_header:
            obj["default_witness_commitment"] = pblocktemplate.signet_header.GetWitnessCommitment().hex()

    return obj


    fPreSegWit = not DeploymentActiveAfter(chainman.ActiveChainstate().m_chain.Tip(), chainman, Consensus.DEPLOYMENT_SEGWIT)

    aCaps = [
        "proposal"
    ]

    transactions = []
    setTxIndex = {}
    i = 0
    for tx in node.blockassembler.block.vtx:
        txHash = tx.GetHash()
        setTxIndex[txHash] = i
        i += 1

        if tx.IsCoinBase():
            continue

        entry = {}

        entry["data"] = EncodeHexTx(tx)
        entry["txid"] = txHash.GetHex()
        entry["hash"] = tx.GetWitnessHash().GetHex()

        deps = []
        for in_ in tx.vin:
            if in_.prevout.hash in setTxIndex:
                deps.append(setTxIndex[in_.prevout.hash])
        entry["depends"] = deps

        index_in_template = i - 1
        entry["fee"] = node.blockassembler.vTxFees[index_in_template]
        nTxSigOps = node.blockassembler.vTxSigOpsCost[index_in_template]
        if fPreSegWit:
            CHECK_NONFATAL(nTxSigOps % WITNESS_SCALE_FACTOR == 0)
            nTxSigOps /= WITNESS_SCALE_FACTOR
        entry["sigops"] = nTxSigOps
        entry["weight"] = GetTransactionWeight(tx)

        transactions.append(entry)

    aux = {}

    hashTarget = arith_uint256().SetCompact(node.blockassembler.block.nBits)

    aMutable = [
        "time",
        "transactions",
        "prevblock"
    ]

    result = {}
    result["capabilities"] = aCaps

    aRules = [
        "csv"
    ]
    if not fPreSegWit:
        aRules.append("!segwit")
    if chainman.GetParams().signet_blocks:
        # indicate to miner that they must understand signet rules
        # when attempting to mine with this template
        aRules.append("!signet")

    vbavailable = {}
    for j in range(0, Consensus.MAX_VERSION_BITS_DEPLOYMENTS):
        pos = Consensus.DeploymentPos(j)
        state = chainman.m_versionbitscache.State(chainman.ActiveChainstate().m_chain.Tip(), chainman.GetParams(), pos)
        switch (state):
            case ThresholdState.DEFINED:
            case ThresholdState.FAILED:
                # Not exposed to GBT at all
                break
            case ThresholdState.LOCKED_IN:
                # Ensure bit is set in block version
                node.blockassembler.block.nVersion |= chainman.m_versionbitscache.Mask(chainman.GetParams(), pos)
                [[fallthrough]]
            case ThresholdState.STARTED:
            {
                vbinfo = VersionBitsDeploymentInfo[pos]
                vbavailable[gbt_vb_name(pos)] = chainman.GetParams().vDeployments[pos].bit
                if vbinfo.name not in setClientRules:
                    if not vbinfo.gbt_force:
                        # If the client doesn't support this, don't indicate it in the [default] version
                        node.blockassembler.block.nVersion &= ~chainman.m_versionbitscache.Mask(chainman.GetParams(), pos)
                    break
                }
            case ThresholdState.ACTIVE:
            {
                # Add to rules only
                vbinfo = VersionBitsDeploymentInfo[pos]
                aRules.append(gbt_vb_name(pos))
                if vbinfo.name not in setClientRules:
                    # Not supported by the client; make sure it's safe to proceed
                    if not vbinfo.gbt_force:
                        raise JSONRPCError(RPC_INVALID_PARAMETER, strprintf("Support for '%s' rule requires explicit client support", vbinfo.name))
                    break
                }
            }
        }
    result["version"] = node.blockassembler.block.nVersion
    result["rules"] = aRules
        result["vbavailable"] = vbavailable
    result["vbrequired"] = int(0)

    result["previousblockhash"] = node.blockassembler.block.hashPrevBlock.hex()
    result["transactions"] = transactions
    result["coinbaseaux"] = aux
    result["coinbasevalue"] = node.blockassembler.block.vtx[0].vout[0].nValue
    result["longpollid"] = chainman.ActiveChainstate().m_chain.Tip().GetBlockHash().hex() + ToString(node.blockassembler.nTransactionsUpdatedLast)
    result["target"] = hashTarget.hex()
    result["mintime"] = node.blockassembler.block.GetMedianTimePast() + 1
    result["mutable"] = aMutable
    result["noncerange"] = "00000000ffffffff"
    int64_t nSigOpLimit = MAX_BLOCK_SIGOPS_COST
    int64_t nSizeLimit = MAX_BLOCK_SERIALIZED_SIZE
    if fPreSegWit:
        CHECK_NONFATAL(nSigOpLimit % WITNESS_SCALE_FACTOR == 0)
        nSigOpLimit /= WITNESS_SCALE_FACTOR
        CHECK_NONFATAL(nSizeLimit % WITNESS_SCALE_FACTOR == 0)
        nSizeLimit /= WITNESS_SCALE_FACTOR
    result["sigoplimit"] = nSigOpLimit
    result["sizelimit"] = nSizeLimit
    if not fPreSegWit:
        result["weightlimit"] = (int64_t)MAX_BLOCK_WEIGHT
    result["curtime"] = node.blockassembler.block.GetBlockTime()
    result["bits"] = strprintf("%08x", node.blockassembler.block.nBits)
    result["height"] = (int64_t)(node.blockassembler.block.nHeight + 1)

    if chainman.GetParams().signet_blocks:
        result["signet_challenge"] = HexStr(chainman.GetParams().signet_challenge)

    if not node.blockassembler.vchCoinbaseCommitment.empty():
        result["default_witness_commitment"] = HexStr(node.blockassembler.vchCoinbaseCommitment)

    return result

class SubmitBlockStateCatcher(CValidationInterface):
    def __init__(self, hash):
        self.hash = hash
        self.found = False
        self.state = BlockValidationState()

    def BlockChecked(self, block, stateIn):
        if block.GetHash() != self.hash:
            return

        self.found = True
        self.state = stateIn

def submitblock(request):
    blockptr = CBlock()
    block = blockptr

    if not DecodeHexBlk(block, request.params[0].get_str()):
        raise JSONRPCError(RPC_DESERIALIZATION_ERROR, "Block decode failed")

    if block.vtx.empty() or not block.vtx[0].IsCoinBase():
        raise JSONRPCError(RPC_DESERIALIZATION_ERROR, "Block does not start with a coinbase")

    chainman = EnsureAnyChainman(request.context)
    hash = block.GetHash()

    with LOCK(cs_main):
        pindex = chainman.m_blockman.LookupBlockIndex(hash)
        if pindex:
            if pindex.IsValid(BLOCK_VALID_SCRIPTS):
                return "duplicate"

            if pindex.nStatus & BLOCK_FAILED_MASK:
                return "duplicate-invalid"

    with LOCK(cs_main):
        pindex = chainman.m_blockman.LookupBlockIndex(block.hashPrevBlock)
        if pindex:
            chainman.UpdateUncommittedBlockStructures(block, pindex)

    new_block = False
    sc = SubmitBlockStateCatcher(block.GetHash())
    RegisterSharedValidationInterface(sc)
    accepted = chainman.ProcessNewBlock(blockptr, force_processing=True, min_pow_checked=True, new_block=&new_block)
    UnregisterSharedValidationInterface(sc)
    if not new_block and accepted:
        return "duplicate"

    if not sc.found:
        return "inconclusive"

    return BIP22ValidationResult(sc.state)

def submitheader(request):
    h = CBlockHeader()

    if not DecodeHexBlockHeader(h, request.params[0].get_str()):
        raise JSONRPCError(RPC_DESERIALIZATION_ERROR, "Block header decode failed")

    chainman = EnsureAnyChainman(request.context)

    with LOCK(cs_main):
        if not chainman.m_blockman.LookupBlockIndex(h.hashPrevBlock):
            raise JSONRPCError(RPC_VERIFY_ERROR, "Must submit previous header (" + h.hashPrevBlock.GetHex() + ") first")

    state = BlockValidationState()
    chainman.ProcessNewBlockHeaders([h], min_pow_checked=True, state=state)
    if state.IsValid():
        return

    if state.IsError():
        raise JSONRPCError(RPC_VERIFY_ERROR, state.ToString())

    raise JSONRPCError(RPC_VERIFY_ERROR, state.GetRejectReason())

def RegisterMiningRPCCommands(t):
    commands = [
        ("mining", getnetworkhashps),
        ("mining", getmininginfo),
        ("mining", prioritisetransaction),
        ("mining", getprioritisedtransactions),
        ("mining", getblocktemplate),
        ("mining", submitblock),
        ("mining", submitheader),

        ("hidden", generatetoaddress),
        ("hidden", generatetodescriptor),
        ("hidden", generateblock),
        ("hidden", generate),
    ]

    for c in commands:
        t.appendCommand(c[0], c[1])
