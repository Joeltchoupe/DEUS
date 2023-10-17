#ABORT NODE
import logging
import node.interface_ui as interface_ui
import shutdown
import util.translation as translation
import warnings

from node.warnings import SetMiscWarning

def AbortNode(exit_status, debug_message, user_message, shutdown=True):
    """ Aborts the node.

    Args:
        exit_status (atomic.int): The exit status to set.
        debug_message (str): The debug message to log.
        user_message (bilingual_str): The user message to display.
        shutdown (bool): Whether to start the shutdown process.

    Returns:
        None
    """

    SetMiscWarning(translation.Untranslated(debug_message))
    logging.error("*** %s\n", debug_message)
    interface_ui.InitError(user_message.empty() and translation._("A fatal internal error occurred, see debug.log for details") or user_message)
    exit_status.store(exit_status)
    if shutdown:
        shutdown.StartShutdown()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#BLOCK MANAGER ARGS

import util.result as result
import util.translation as translation
import validation

def apply_args_man_options(args: ArgsManager, opts: BlockManager.Options) -> result.Result[None]:
    """Apply the block manager arguments to the given options.

    Args:
        args (ArgsManager): The arguments manager.
        opts (BlockManager.Options): The options to apply the arguments to.

    Returns:
        result.Result[None]: An empty result if successful, or an error otherwise.
    """

    # Block pruning; get the amount of disk space (in MiB) to allot for block & undo files.
    prune_arg = args.get_int_arg("-prune", opts.prune_target)
    if prune_arg < 0:
        return result.Error(translation._("Prune cannot be configured with a negative value."))

    prune_target = uint64(prune_arg) * 1024 * 1024
    if prune_arg == 1:  # Manual pruning: -prune=1
        prune_target = BlockManager.PRUNE_TARGET_MANUAL
    elif prune_target:
        if prune_target < validation.MIN_DISK_SPACE_FOR_BLOCK_FILES:
            return result.Error(
                strprintf(
                    translation._(
                        "Prune configured below the minimum of %d MiB.  Please use a higher number."
                    ),
                    validation.MIN_DISK_SPACE_FOR_BLOCK_FILES / 1024 / 1024,
                )
            )

    opts.prune_target = prune_target

    if "fastprune" in args.arguments:
        opts.fast_prune = args.get_bool_arg("-fastprune")

    return result.Ok()
#-----------------------------------------------------------------------------------------------------
#CACHES
import util.args as args
import index.txindex as txindex

def calculate_cache_sizes(args: ArgsManager, n_indexes: int) -> CacheSizes:
    """Calculates the cache sizes for the given arguments and number of indexes.

    Args:
        args (ArgsManager): The arguments manager.
        n_indexes (int): The number of indexes.

    Returns:
        CacheSizes: The calculated cache sizes.
    """

    total_cache = args.get_int_arg("-dbcache", args.DEFAULT_DB_CACHE) << 20
    total_cache = max(total_cache, args.N_MIN_DB_CACHE << 20)
    total_cache = min(total_cache, args.N_MAX_DB_CACHE << 20)

    sizes = CacheSizes()
    sizes.block_tree_db = min(total_cache // 8, args.N_MAX_BLOCK_DB_CACHE << 20)
    total_cache -= sizes.block_tree_db
    sizes.tx_index = min(total_cache // 8, args.get_bool_arg("-txindex", txindex.DEFAULT_TXINDEX) and args.N_MAX_TX_INDEX_CACHE << 20 or 0)
    total_cache -= sizes.tx_index
    sizes.filter_index = 0
    if n_indexes > 0:
        max_cache = min(total_cache // 8, args.N_MAX_FILTER_INDEX_CACHE << 20)
        sizes.filter_index = max_cache // n_indexes
        total_cache -= sizes.filter_index * n_indexes
    sizes.coins_db = min(total_cache // 2, (total_cache // 4) + (1 << 23))
    sizes.coins_db = min(sizes.coins_db, args.N_MAX_COINS_DB_CACHE << 20)
    total_cache -= sizes.coins_db
    sizes.coins = total_cache

    return sizes

class CacheSizes:
    """Represents the cache sizes."""

    def __init__(self):
        self.block_tree_db = 0
        self.tx_index = 0
        self.filter_index = 0
        self.coins_db = 0
        self.coins = 0

#-----------------------------------------------------------------------------------------------------
#CHAINSTATE
import util.args as args
import util.result as result
import util.strencodings as strencodings
import util.translation as translation
import validation

def apply_args_man_options(args: ArgsManager, opts: ChainstateManager.Options) -> result.Result[None]:
    """Apply the chainstate manager arguments to the given options.

    Args:
        args (ArgsManager): The arguments manager.
        opts (ChainstateManager.Options): The options to apply the arguments to.

    Returns:
        result.Result[None]: An empty result if successful, or an error otherwise.
    """

    if "checkblockindex" in args.arguments:
        opts.check_block_index = args.get_bool_arg("-checkblockindex")

    if "checkpoints" in args.arguments:
        opts.checkpoints_enabled = args.get_bool_arg("-checkpoints")

    if "minimumchainwork" in args.arguments:
        minimum_chain_work = args.get_arg("-minimumchainwork")
        if not strencodings.IsHexNumber(minimum_chain_work):
            return result.Error(translation._("Invalid non-hex (%s) minimum chain work value specified") % minimum_chain_work)
        opts.minimum_chain_work = strencodings.UintToArith256(strencodings.uint256S(minimum_chain_work))

    if "assumevalid" in args.arguments:
        opts.assumed_valid_block = strencodings.uint256S(args.get_arg("-assumevalid"))

    if "maxtipage" in args.arguments:
        opts.max_tip_age = std::chrono::seconds(args.get_int_arg("-maxtipage"))

    read_database_args(args, opts.block_tree_db)
    read_database_args(args, opts.coins_db)
    read_coins_view_args(args, opts.coins_view)

    return result.Ok()

def read_database_args(args: ArgsManager, opts: DatabaseArgs) -> None:
    """Read the database arguments from the given arguments manager.

    Args:
        args (ArgsManager): The arguments manager.
        opts (DatabaseArgs): The options to read the database arguments into.
    """

    if "dbcache" in args.arguments:
        opts.cache_size_bytes = args.get_int_arg("-dbcache", args.DEFAULT_DB_CACHE) << 20

    if "dblogsize" in args.arguments:
        opts.log_size_bytes = args.get_int_arg("-dblogsize", args.DEFAULT_DB_LOG_SIZE) << 20

    if "flushwallet" in args.arguments:
        opts.flush_wallet = args.get_bool_arg("-flushwallet")

    if "usehd" in args.arguments:
        opts.use_hd = args.get_bool_arg("-usehd")

    if "walletdb" in args.arguments:
        opts.wallet_db_path = args.get_arg("-walletdb")

    if "walletprefix" in args.arguments:
        opts.wallet_prefix = args.get_arg("-walletprefix")

def read_coins_view_args(args: ArgsManager, opts: CoinsViewArgs) -> None:
    """Read the coins view arguments from the given arguments manager.

    Args:
        args (ArgsManager): The arguments manager.
        opts (CoinsViewArgs): The options to read the coins view arguments into.
    """

    if "txindex" in args.arguments:
        opts.txindex_enabled = args.get_bool_arg("-txindex", validation.DEFAULT_TXINDEX)

    if "timestampindex" in args.arguments:
        opts.timestampindex_enabled = args.get_bool_arg("-timestampindex")

    if "addressindex" in args.arguments:
        opts.addressindex_enabled = args.get_bool_arg("-addressindex", validation.DEFAULT_ADDRESSINDEX)

    if "spentindex" in args.arguments:
        opts.spentindex_enabled = args.get_bool_arg("-spentindex", validation.DEFAULT_SPENTINDEX)

    if "balancesindex" in args.arguments:
        opts.balancesindex_enabled = args.get_bool_arg("-balancesindex", validation.DEFAULT_BALANCESINDEX)

#---------------------------------------------------------------------------------------------------------------------------------------------
#COIN

import #node? VERIFY IT
import txmempool
import validation

def find_coins(node_context: node.NodeContext, coins: dict[COutPoint, Coin]) -> None:
    """Finds the given coins in the chainstate and mempool.

    Args:
        node_context (node.NodeContext): The node context.
        coins (dict[COutPoint, Coin]): The coins to find.
    """

    assert node_context.mempool is not None
    assert node_context.chainman is not None

    with node_context.mempool.cs, node_context.chainman.ActiveChainstate().CoinsTip().cs:
        mempool_view = txmempool.CCoinsViewMemPool(node_context.chainman.ActiveChainstate().CoinsTip(), node_context.mempool)

        for outpoint, coin in coins.items():
            if not mempool_view.GetCoin(outpoint, coin):
                # Either the coin is not in the CCoinsViewCache or is spent. Clear it.
                coin.Clear()
#----------------------------------------------------------------------------------------------------------------------------------------------------
import util.args as args

def read_coins_view_args(args: ArgsManager, options: CoinsViewOptions) -> None:
    """Reads the coins view arguments from the given arguments manager.

    Args:
        args (ArgsManager): The arguments manager.
        options (CoinsViewOptions): The options to read the coins view arguments into.
    """

    if "dbbatchsize" in args.arguments:
        options.batch_write_bytes = args.get_int_arg("-dbbatchsize")

    if "dbcrashratio" in args.arguments:
        options.simulate_crash_ratio = args.get_int_arg("-dbcrashratio")

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def connection_type_as_string(conn_type: ConnectionType) -> str:
    """Converts a connection type to a string.

    Args:
        conn_type (ConnectionType): The connection type.

    Returns:
        str: The connection type as a string.
    """

    if conn_type == ConnectionType.INBOUND:
        return "inbound"
    elif conn_type == ConnectionType.MANUAL:
        return "manual"
    elif conn_type == ConnectionType.FEELER:
        return "feeler"
    elif conn_type == ConnectionType.OUTBOUND_FULL_RELAY:
        return "outbound-full-relay"
    elif conn_type == ConnectionType.BLOCK_RELAY:
        return "block-relay-only"
    elif conn_type == ConnectionType.ADDR_FETCH:
        return "addr-fetch"
    else:
        assert False, "Unknown connection type"


def transport_type_as_string(transport_type: TransportProtocolType) -> str:
    """Converts a transport type to a string.

    Args:
        transport_type (TransportProtocolType): The transport type.

    Returns:
        str: The transport type as a string.
    """

    if transport_type == TransportProtocolType.DETECTING:
        return "detecting"
    elif transport_type == TransportProtocolType.V1:
        return "v1"
    elif transport_type == TransportProtocolType.V2:
        return "v2"
    else:
        assert False, "Unknown transport type"
#----------------------------------------------------------------------------------------------------------------

class NodeContext:
    """A node context contains all the information needed to run a Bitcoin node."""

    def __init__(self):
        self.chain = interfaces.Chain()
        self.kernel = kernel.Context()
        self.net = net.CNet()
        self.net_processing = net_processing.CConnman()
        self.netgroup = netgroup.CNetGroup()
        self.addrman = addrman.CAddrMan()
        self.banman = banman.BanMan()
        self.txmempool = txmempool.CTxMemPool()
        self.scheduler = scheduler.CScheduler()
        self.kernel_notifications = node.KernelNotifications()
        self.policy_fees = policy.FeeFilter()

#--------------------------------------------------------------------------------------------------------------------

import util.args as args

def read_database_args(args: ArgsManager, options: DBOptions) -> None:
    """Reads the database arguments from the given arguments manager.

    Args:
        args (ArgsManager): The arguments manager.
        options (DBOptions): The options to read the database arguments into.
    """

    if "forcecompactdb" in args.arguments:
        options.force_compact = args.get_bool_arg("-forcecompactdb")

#----------------------------------------------------------------------------------------------------------------------

import node
import util.args as args
import util.strencodings as strencodings

class KernelNotifications:
    def __init__(self, exit_status, shutdown_on_fatal_error):
        self.m_exit_status = exit_status
        self.m_shutdown_on_fatal_error = shutdown_on_fatal_error
        self.m_stop_at_height = None

    def block_tip(self, state, index):
        node.ui_interface.NotifyBlockTip(state, index)
        if self.m_stop_at_height and index.nHeight >= self.m_stop_at_height:
            node.StartShutdown()
            return node.kernel.Interrupted()
        return {}

    def header_tip(self, state, height, timestamp, presync):
        node.ui_interface.NotifyHeaderTip(state, height, timestamp, presync)

    def progress(self, title, progress_percent, resume_possible):
        node.ui_interface.ShowProgress(title.translated, progress_percent, resume_possible)

    def warning(self, warning):
        self.do_warning(warning)

    def flush_error(self, debug_message):
        node.AbortNode(self.m_exit_status, debug_message)

    def fatal_error(self, debug_message, user_message):
        node.AbortNode(self.m_exit_status, debug_message, user_message, self.m_shutdown_on_fatal_error)

    def do_warning(self, warning):
        static f_warned = False
        node.SetMiscWarning(warning)
        if not f_warned:
            node.AlertNotify(warning.original)
            f_warned = True

def read_notification_args(args: ArgsManager, notifications: KernelNotifications) -> None:
    """Reads the notification arguments from the given arguments manager.

    Args:
        args (ArgsManager): The arguments manager.
        notifications (KernelNotifications): The notifications to read the arguments into.
    """

    if "stopatheight" in args.arguments:
        notifications.m_stop_at_height = args.get_int_arg("-stopatheight")

#---------------------------------------------------------------------------------------------------------------------

import decimal
import util.args as args
import util.moneystr as moneystr
import util.translation as translation

def apply_args_man_options(argsman: ArgsManager, mempool_limits: MemPoolLimits) -> None:
    """Applies the arguments manager options to the mempool limits.

    Args:
        argsman (ArgsManager): The arguments manager.
        mempool_limits (MemPoolLimits): The mempool limits.
    """

    mempool_limits.ancestor_count = argsman.get_int_arg("-limitancestorcount", mempool_limits.ancestor_count)

    if "limitancestorsize" in argsman.arguments:
        mempool_limits.ancestor_size_vbytes = argsman.get_int_arg("-limitancestorsize") * 1000

    mempool_limits.descendant_count = argsman.get_int_arg("-limitdescendantcount", mempool_limits.descendant_count)

    if "limitdescendantsize" in argsman.arguments:
        mempool_limits.descendant_size_vbytes = argsman.get_int_arg("-limitdescendantsize") * 1000

def apply_args_man_options(argsman: ArgsManager, chainparams: CChainParams, mempool_opts: MemPoolOptions) -> util.Result[void]:
    """Applies the arguments manager options to the mempool options.

    Args:
        argsman (ArgsManager): The arguments manager.
        chainparams (CChainParams): The chain parameters.
        mempool_opts (MemPoolOptions): The mempool options.

    Returns:
        util.Result[void]: None if the options were applied successfully, otherwise an error.
    """

    mempool_opts.check_ratio = argsman.get_int_arg("-checkmempool", mempool_opts.check_ratio)

    if "maxmempool" in argsman.arguments:
        mempool_opts.max_size_bytes = argsman.get_int_arg("-maxmempool") * 1000000

    if "mempoolexpiry" in argsman.arguments:
        mempool_opts.expiry = decimal.Decimal(argsman.get_int_arg("-mempoolexpiry")) * 3600

    # incremental relay fee sets the minimum feerate increase necessary for replacement in the mempool
    # and the amount the mempool min fee increases above the feerate of txs evicted due to mempool limiting.
    if "incrementalrelayfee" in argsman.arguments:
        try:
            mempool_opts.incremental_relay_feerate = CFeeRate(decimal.Decimal(argsman.get_arg("-incrementalrelayfee")))
        except ValueError:
            return util.Error(_("Invalid incrementalrelayfee: {}").format(argsman.get_arg("-incrementalrelayfee")))

    if "minrelaytxfee" in argsman.arguments:
        try:
            mempool_opts.min_relay_feerate = CFeeRate(decimal.Decimal(argsman.get_arg("-minrelaytxfee")))
        except ValueError:
            return util.Error(_("Invalid minrelaytxfee: {}").format(argsman.get_arg("-minrelaytxfee")))
    else:
        if mempool_opts.incremental_relay_feerate > mempool_opts.min_relay_feerate:
            # Allow only setting incremental fee to control both
            mempool_opts.min_relay_feerate = mempool_opts.incremental_relay_feerate
            print(_("Increasing minrelaytxfee to {} to match incrementalrelayfee\n").format(mempool_opts.min_relay_feerate.to_string()))

    # Feerate used to define dust.  Shouldn't be changed lightly as old
    # implementations may inadvertently create non-standard transactions
    if "dustrelayfee" in argsman.arguments:
        try:
            mempool_opts.dust_relay_feerate = CFeeRate(decimal.Decimal(argsman.get_arg("-dustrelayfee")))
        except ValueError:
            return util.Error(_("Invalid dustrelayfee: {}").format(argsman.get_arg("-dustrelayfee")))

    mempool_opts.permit_bare_multisig = argsman.get_bool_arg("-permitbaremultisig", DEFAULT_PERMIT_BAREMULTISIG)

    if "datacarrier" in argsman.arguments:
        mempool_opts.max_datacarrier_bytes = argsman.get_int_arg("-datacarriersize", MAX_OP_RETURN_RELAY)
    else:
        mempool_opts.max_datacarrier_bytes = None

    mempool_opts.require_standard = not argsman.get_bool_arg("-acceptnonstdtxn", DEFAULT_ACCEPT_NON_STD_TXN)
    if not chainparams.IsTestChain() and not mempool_opts.require_standard:
        return util.Error(_("acceptnonstdtxn is not currently supported for {} chain").format(chainparams.GetChainTypeString()))

    mempool_opts.full_rbf = argsman.get_bool_arg("-mempoolfullrbf", mempool_opts.full_rbf)

    apply_args_man_options(argsman, mempool_opts.limits)

    return None

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

import util.args as args
import util.fs as fs

def should_persist_mempool(argsman: ArgsManager) -> bool:
    """Returns True if the mempool should be persisted, False otherwise.

    Args:
        argsman (ArgsManager): The arguments manager.

    Returns:
        bool: True if the mempool should be persisted, False otherwise.
    """

    return argsman.get_bool_arg("-persistmempool", DEFAULT_PERSIST_MEMPOOL)

def mempool_path(argsman: ArgsManager) -> fs.path:
    """Returns the path to the mempool data file.

    Args:
        argsman (ArgsManager): The arguments manager.

    Returns:
        fs.path: The path to the mempool data file.
    """

    return argsman.get_datadir_net() / "mempool.dat"

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

import logging
import math
from time import time
from typing import Optional, Tuple

from minisketch import Minisketch

# We can't use `logging.getLogger(__name__)` here because this is a submodule
logger = logging.getLogger('node.minisketchwrapper')

BITS = 32

def find_best_implementation() -> int:
    """Find the best Minisketch implementation for the current platform.

    Returns:
        int: The index of the best implementation.
    """

    best: Optional[Tuple[float, int]] = None

    max_impl = Minisketch.MaxImplementation()
    for impl in range(max_impl + 1):
        if not Minisketch.ImplementationSupported(BITS, impl):
            continue

        benches = []
        offset = 0
        # Run a little benchmark with capacity 32, adding 184 entries, and decoding 11 of them once.
        for b in range(11):
            sketch = Minisketch(BITS, impl, 32)
            start = time()
            for e in range(100):
                sketch.Add(e * 1337 + b * 13337 + offset)
            for e in range(84):
                sketch.Add(e * 1337 + b * 13337 + offset)
            offset += sketch.Decode(32)[0]
            stop = time()
            benches.append(stop - start)

        # Remember which implementation has the best median benchmark time.
        if benches:
            benches.sort()
            if best is None or best[0] > benches[5]:
                best = benches[5], impl

    assert best is not None
    logger.info("Using Minisketch implementation number %i", best[1])
    return best[1]

def minisketch32_implementation() -> int:
    """Find the best Minisketch implementation for the current platform, cached.

    Returns:
        int: The index of the best implementation.
    """

    # Fast compute-once idiom.
    best = find_best_implementation()
    return best

def make_minisketch32(capacity: int) -> Minisketch:
    """Create a new Minisketch object with the given capacity.

    Args:
        capacity: The capacity of the Minisketch object.

    Returns:
        Minisketch: A new Minisketch object.
    """

    return Minisketch(BITS, minisketch32_implementation(), capacity)

def make_minisketch32_fp(max_elements: int, fpbits: int) -> Minisketch:
    """Create a new Minisketch fingerprint object with the given parameters.

    Args:
        max_elements: The maximum number of elements that the Minisketch object will store.
        fpbits: The number of bits in the fingerprint.

    Returns:
        Minisketch: A new Minisketch fingerprint object.
    """

    return Minisketch.CreateFP(BITS, minisketch32_implementation(), max_elements, fpbits)

#---------------------------------------------------------------------------------------------------------------------------------------------

import math
from typing import Optional

from util.args import ArgsManager

from net_processing import PeerManager, PeerManagerOptions

def apply_args_man_options(argsman: ArgsManager, options: PeerManagerOptions) -> None:
    """Applies the arguments manager options to the peer manager options.

    Args:
        argsman (ArgsManager): The arguments manager.
        options (PeerManagerOptions): The peer manager options.
    """

    if "txreconciliation" in argsman.arguments:
        options.reconcile_txs = argsman.get_bool_arg("-txreconciliation")

    if "maxorphantx" in argsman.arguments:
        options.max_orphan_txs = math.clamp(argsman.get_int_arg("-maxorphantx"), 0, 2**32 - 1)

    if "blockreconstructionextratxn" in argsman.arguments:
        options.max_extra_txs = math.clamp(argsman.get_int_arg("-blockreconstructionextratxn"), 0, 2**32 - 1)

    if "capturemessages" in argsman.arguments:
        options.capture_messages = argsman.get_bool_arg("-capturemessages")

    if "blocksonly" in argsman.arguments:
        options.ignore_incoming_txs = argsman.get_bool_arg("-blocksonly")

#--------------------------------------------------------------------------------------------------------------------------------------

import asyncio
from typing import Optional

from block import BlockManager
from chain import ChainstateManager
from mempool import Mempool
from node import NodeContext
from transaction import CTransaction, TransactionError

async def broadcast_transaction(node: NodeContext, tx: CTransaction, max_tx_fee: float, relay: bool, wait_callback: bool) -> TransactionError:
    """Broadcasts a transaction to the network.

    Args:
        node (NodeContext): The node context.
        tx (CTransaction): The transaction to broadcast.
        max_tx_fee (float): The maximum transaction fee to pay.
        relay (bool): Whether to relay the transaction to other nodes.
        wait_callback (bool): Whether to wait for the transaction to be accepted by the mempool before returning.

    Returns:
        TransactionError: The error code, if any.
    """

    chainman = node.chainman
    mempool = node.mempool
    peerman = node.peerman

    assert chainman is not None
    assert mempool is not None
    assert peerman is not None

    txid = tx.GetHash()
    wtxid = tx.GetWitnessHash()
    callback_set = False

    async with chainman.lock:
        # If the transaction is already confirmed in the chain, don't do anything and return early.
        view = chainman.ActiveChainstate().CoinsTip()
        for o in range(len(tx.vout)):
            existing_coin = view.AccessCoin(COutPoint(txid, o))
            # IsSpent doesn't mean the coin is spent, it means the output doesn't exist.
            # So if the output does exist, then this transaction exists in the chain.
            if not existing_coin.IsSpent():
                return TransactionError.ALREADY_IN_CHAIN

        # Check if the transaction is already in the mempool.
        mempool_tx = mempool.get(txid)
        if mempool_tx:
            # The transaction is already in the mempool. Don't try to submit it to the mempool again,
            # but do attempt to reannounce the mempool transaction if relay=true.
            wtxid = mempool_tx.GetWitnessHash()
        else:
            # The transaction is not in the mempool.
            if max_tx_fee > 0:
                # First, call ATMP with test_accept and check the fee. If ATMP fails here,
                # return error immediately.
                result = chainman.ProcessTransaction(tx, test_accept=True)
                if result.m_result_type != MempoolAcceptResult.ResultType.VALID:
                    return TransactionError.MEMPOOL_REJECTED
                elif result.m_base_fees.value() > max_tx_fee:
                    return TransactionError.MAX_FEE_EXCEEDED

            # Try to submit the transaction to the mempool.
            result = chainman.ProcessTransaction(tx, test_accept=False)
            if result.m_result_type != MempoolAcceptResult.ResultType.VALID:
                return TransactionError.MEMPOOL_REJECTED

            # The transaction was accepted to the mempool.

            if relay:
                # The mempool tracks locally submitted transactions to make a best-effort of initial broadcast.
                mempool.AddUnbroadcastTx(txid)

            if wait_callback:
                # For transactions broadcast from outside the wallet, make sure that the wallet has been notified of the
                # transaction before continuing.
                #
                # This prevents a race where a user might call sendrawtransaction with a transaction to/from their wallet,
                # immediately call some wallet RPC, and get a stale result because callbacks have not yet been processed.
                await node.wallet_loader.CallFunctionInValidationInterfaceQueue(lambda: asyncio.sleep(0))
                callback_set = True

    if callback_set:
        # Wait until the wallet has been notified of the transaction entering the mempool.
        await asyncio.sleep(0)

    if relay:
        peerman.RelayTransaction(txid, wtxid)

    return TransactionError.OK

def get_transaction(block_index: Optional[BlockIndex], mempool: Optional[Mempool], hash: bytes, hash_block: bytes, blockman: BlockManager) -> Optional[CTransaction]:
    """Returns the transaction with the given hash.

    Args:
        block_index"""
        
    # Check if the transaction is in the mempool.
    if mempool:
        tx = mempool.get(hash)
        if tx:
            return tx

    # Check if the transaction is in the txindex.
    if txindex:
        tx, hash_block = txindex.FindTx(hash)
        if tx:
            return tx

    # Check if the transaction is in the block.
    if block_index:
        block = blockman.ReadBlockFromDisk(block_index)
        for tx in block.vtx:
            if tx.GetHash() == hash:
                return tx

    return None

#------------------------------------------------------------------------------------------------------------------------------------------

from . import psbt

def analyze_psbt(psbtx: psbt.PartiallySignedTransaction) -> psbt.PSBTAnalysis:
    """Analyzes a PSBT and returns a PSBTAnalysis object.

    Args:
        psbtx (psbt.PartiallySignedTransaction): The PSBT to analyze.

    Returns:
        psbt.PSBTAnalysis: A PSBTAnalysis object.
    """

    result = psbt.PSBTAnalysis()

    calc_fee = True

    in_amt = 0

    result.inputs.resize(len(psbtx.tx.vin))

    txdata = psbt.PrecomputePSBTData(psbtx)

    for i in range(len(psbtx.tx.vin)):
        input = psbtx.inputs[i]
        input_analysis = result.inputs[i]

        # We set next role here and ratchet backwards as required
        input_analysis.next = psbt.PSBTRole.EXTRACTOR

        # Check for a UTXO
        utxo = psbtx.GetInputUTXO(i)
        if utxo:
            if not MoneyRange(utxo.nValue) or not MoneyRange(in_amt + utxo.nValue):
                result.SetInvalid(f"PSBT is not valid. Input {i} has invalid value")
                return result

            in_amt += utxo.nValue
            input_analysis.has_utxo = True
        else:
            if input.non_witness_utxo and psbtx.tx.vin[i].prevout.n >= len(input.non_witness_utxo.vout):
                result.SetInvalid(f"PSBT is not valid. Input {i} specifies invalid prevout")
                return result

            input_analysis.has_utxo = False
            input_analysis.is_final = False
            input_analysis.next = psbt.PSBTRole.UPDATER
            calc_fee = False

        if utxo and utxo.scriptPubKey.IsUnspendable():
            result.SetInvalid(f"PSBT is not valid. Input {i} spends unspendable output")
            return result

        # Check if it is final
        if not psbt.PSBTInputSignedAndVerified(psbtx, i, txdata):
            input_analysis.is_final = False

            # Figure out what is missing
            outdata = psbt.SignatureData()
            complete = psbt.SignPSBTInput(psbtx, i, txdata, 1, outdata)

            # Things are missing
            if not complete:
                input_analysis.missing_pubkeys = outdata.missing_pubkeys
                input_analysis.missing_redeem_script = outdata.missing_redeem_script
                input_analysis.missing_witness_script = outdata.missing_witness_script
                input_analysis.missing_sigs = outdata.missing_sigs

                # If we are only missing signatures and nothing else, then next is signer
                if not outdata.missing_pubkeys and outdata.missing_redeem_script is None and outdata.missing_witness_script is None and len(outdata.missing_sigs) > 0:
                    input_analysis.next = psbt.PSBTRole.SIGNER
                else:
                    input_analysis.next = psbt.PSBTRole.UPDATER
            else:
                input_analysis.next = psbt.PSBTRole.FINALIZER
        elif utxo:
            input_analysis.is_final = True

    # Calculate next role for PSBT by grabbing "minimum" PSBTInput next role
    result.next = psbt.PSBTRole.EXTRACTOR
    for input_analysis in result.inputs:
        result.next = min(result.next, input_analysis.next)
    assert result.next > psbt.PSBTRole.CREATOR

    if calc_fee:
        # Get the output amount
        out_amt = sum(txout.nValue for txout in psbtx.tx.vout)
        if not MoneyRange(out_amt):
            result.SetInvalid("PSBT is not valid. Output amount invalid")
            return result

        # Get the fee
        fee = in_amt - out_amt
        result.fee = fee

        # Estimate the size
                # Estimate the size
        mtx = psbtx.tx.copy()
        view = CCoinsView()
        success = True

        for i in range(len(psbtx.tx.vin)):
            input = psbtx.inputs[i]
            coin = Coin()

            if not psbt.SignPSBTInput(DUMMY_SIGNING_PROVIDER, psbtx, i, None, 1) or not psbtx.GetInputUTXO(coin.out, i):
                success = False
                break
            else:
                mtx.vin[i].scriptSig = input.final_script_sig
                mtx.vin[i].scriptWitness = input.final_script_witness
                coin.nHeight = 1
                view.AddCoin(psbtx.tx.vin[i].prevout, std::move(coin), True)

        if success:
            size = GetVirtualTransactionSize(mtx, GetTransactionSigOpCost(mtx, view, STANDARD_SCRIPT_VERIFY_FLAGS), ::nBytesPerSigOp)
            result.estimated_vsize = size
            # Estimate fee rate
            feerate = CFeeRate(fee, size)
            result.estimated_feerate = feerate

    return result

#-------------------------------------------------------------------------------------------------------------------------------------------------

class TxReconciliationTracker:
    def __init__(self, recon_version: int):
        self._impl = TxReconciliationTrackerImpl(recon_version)

    def pre_register_peer(self, peer_id: NodeId) -> int:
        return self._impl.pre_register_peer(peer_id)

    def register_peer(self, peer_id: NodeId, is_peer_inbound: bool, peer_recon_version: int, remote_salt: int) -> ReconciliationRegisterResult:
        return self._impl.register_peer(peer_id, is_peer_inbound, peer_recon_version, remote_salt)

    def forget_peer(self, peer_id: NodeId):
        self._impl.forget_peer(peer_id)

    def is_peer_registered(self, peer_id: NodeId) -> bool:
        return self._impl.is_peer_registered(peer_id)


class TxReconciliationTrackerImpl:
    def __init__(self, recon_version: int):
        self._lock = Mutex()
        self._recon_version = recon_version
        self._states = {}

    def pre_register_peer(self, peer_id: NodeId) -> int:
        with self._lock:
            local_salt = GetRand(UINT64_MAX)
            self._states[peer_id] = local_salt
            return local_salt

    def register_peer(self, peer_id: NodeId, is_peer_inbound: bool, peer_recon_version: int, remote_salt: int) -> ReconciliationRegisterResult:
        with self._lock:
            recon_state = self._states.get(peer_id)
            if recon_state is None:
                return ReconciliationRegisterResult.NOT_FOUND

            if isinstance(recon_state, TxReconciliationState):
                return ReconciliationRegisterResult.ALREADY_REGISTERED

            local_salt = recon_state

            # If the peer supports a version which is lower than ours, we downgrade to the version
            # it supports. For now, this only guarantees that nodes with future reconciliation
            # versions have the choice of reconciling with this current version. However, they also
            # have the choice to refuse supporting reconciliations if the common version is not
            # satisfactory (e.g. too low).
            recon_version = min(peer_recon_version, self._recon_version)
            # v1 is the lowest version, so suggesting something below must be a protocol violation.
            if recon_version < 1:
                return ReconciliationRegisterResult.PROTOCOL_VIOLATION

            full_salt = ComputeSalt(local_salt, remote_salt)
            recon_state = TxReconciliationState(!is_peer_inbound, full_salt.GetUint64(0), full_salt.GetUint64(1))
            self._states[peer_id] = recon_state
            return ReconciliationRegisterResult.SUCCESS

    def forget_peer(self, peer_id: NodeId):
        with self._lock:
            if peer_id in self._states:
                del self._states[peer_id]

    def is_peer_registered(self, peer_id: NodeId) -> bool:
        with self._lock:
            recon_state = self._states.get(peer_id)
            return isinstance(recon_state, TxReconciliationState)

#-------------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
from typing import Optional

import util.fs as fs

from .chainstate import Chainstate
from .coinsdb import CoinsDB

SNAPSHOT_BLOCKHASH_FILENAME = "snapshot_blockhash"

def write_snapshot_base_blockhash(snapshot_chainstate: Chainstate) -> bool:
    """Writes the base blockhash of the snapshot chainstate to a file.

    Args:
        snapshot_chainstate: The snapshot chainstate.

    Returns:
        True if the write was successful, False otherwise.
    """

    assert snapshot_chainstate.m_from_snapshot_blockhash is not None

    chaindir = snapshot_chainstate.CoinsDB().StoragePath()
    assert chaindir is not None

    write_to = os.path.join(chaindir, SNAPSHOT_BLOCKHASH_FILENAME)

    with open(write_to, "wb") as f:
        f.write(snapshot_chainstate.m_from_snapshot_blockhash.ToBytes())

    return True

def read_snapshot_base_blockhash(chaindir: str) -> Optional[bytes]:
    """Reads the base blockhash of the snapshot chainstate from a file.

    Args:
        chaindir: The directory containing the snapshot chainstate.

    Returns:
        The base blockhash of the snapshot chainstate, or None if it could not be read.
    """

    if not os.path.exists(chaindir):
        logging.error("cannot read base blockhash: no chainstate dir exists at path %s\n", chaindir)
        return None

    read_from = os.path.join(chaindir, SNAPSHOT_BLOCKHASH_FILENAME)

    if not os.path.exists(read_from):
        logging.error("snapshot chainstate dir is malformed! no base blockhash file "
            "exists at path %s. Try deleting %s and calling loadtxoutset again?\n",
            chaindir, read_from)
        return None

    base_blockhash = bytearray()

    with open(read_from, "rb") as f:
        base_blockhash += f.read()

    return base_blockhash

def find_snapshot_chainstate_dir(data_dir: str) -> Optional[str]:
    """Finds the directory containing the snapshot chainstate.

    Args:
        data_dir: The data directory.

    Returns:
        The directory containing the snapshot chainstate, or None if it could not be found.
    """

    possible_dir = os.path.join(data_dir, f"chainstate{SNAPSHOT_CHAINSTATE_SUFFIX}")

    if os.path.exists(possible_dir):
        return possible_dir

    return None

#-------------------------------------------------------------------------------------------------------------------------------------------

import argparse

from kernel.validation_cache_sizes import ValidationCacheSizes

def apply_args_man_options(argsman: argparse.ArgumentParser, cache_sizes: ValidationCacheSizes) -> None:
    """Applies the arguments from the given argument manager to the validation cache sizes.

    Args:
        argsman: The argument manager.
        cache_sizes: The validation cache sizes.
    """

    if argsman.get("maxsigcachesize"):
        max_size = argsman.get_int("maxsigcachesize")
        # 1. When supplied with a max_size of 0, both InitSignatureCache and
        #    InitScriptExecutionCache create the minimum possible cache (2
        #    elements). Therefore, we can use 0 as a floor here.
        # 2. Multiply first, divide after to avoid integer truncation.
        clamped_size_each = max(max_size, 0) * (1 << 20) // 2
        cache_sizes = ValidationCacheSizes(
            signature_cache_bytes=clamped_size_each,
            script_execution_cache_bytes=clamped_size_each,
        )


