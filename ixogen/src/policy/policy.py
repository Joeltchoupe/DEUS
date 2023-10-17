class CFeeRate:
    def __init__(self, nFeePaid: int, num_bytes: int):
        self.nSatoshisPerK = nFeePaid * 1000 // num_bytes if num_bytes > 0 else 0

    def GetFee(self, num_bytes: int) -> int:
        nFee = int(math.ceil(self.nSatoshisPerK * num_bytes / 1000.0))
        if nFee == 0 and num_bytes != 0:
            if self.nSatoshisPerK > 0:
                nFee = 1
            elif self.nSatoshisPerK < 0:
                nFee = -1
        return nFee

    def ToString(self, fee_estimate_mode: FeeEstimateMode) -> str:
        if fee_estimate_mode == FeeEstimateMode.SAT_VB:
            return f"{self.nSatoshisPerK // 1000}.{self.nSatoshisPerK % 1000} {CURRENCY_ATOM}/vB"
        else:
            return f"{self.nSatoshisPerK // COIN}.{self.nSatoshisPerK % COIN} {CURRENCY_UNIT}/kvB"

#--------------------------------------------------------------------------------------------

import os

FEE_ESTIMATES_FILENAME = "fee_estimates.dat"

def feeest_path(argsman: argparse.ArgumentParser) -> str:
  """Returns the path to the fee estimates file.

  Args:
    argsman: The argument manager.

  Returns:
    The path to the fee estimates file.
  """

  return os.path.join(argsman.get_datadir(), FEE_ESTIMATES_FILENAME)
#---------------------------------------------------------------------------------------------------

from typing import List, Set

from primitives.transaction import Transaction
from util.hasher import SaltedTxidHasher, SaltedOutpointHasher

MAX_PACKAGE_COUNT = 100
MAX_PACKAGE_WEIGHT = 1000000

def check_package(txns: List[Transaction], state: PackageValidationState) -> bool:
    """Checks if the given package is valid.

    Args:
        txns: The package of transactions.
        state: The package validation state.

    Returns:
        True if the package is valid, False otherwise.
    """

    package_count = len(txns)

    if package_count > MAX_PACKAGE_COUNT:
        state.invalid(PackageValidationResult.PCKG_POLICY, "package-too-many-transactions")
        return False

    total_weight = sum([get_transaction_weight(tx) for tx in txns])
    # If the package only contains 1 tx, it's better to report the policy violation on individual tx weight.
    if package_count > 1 and total_weight > MAX_PACKAGE_WEIGHT:
        state.invalid(PackageValidationResult.PCKG_POLICY, "package-too-large")
        return False

    # Require the package to be sorted in order of dependency, i.e. parents appear before children.
    # An unsorted package will fail anyway on missing-inputs, but it's better to quit earlier and
    # fail on something less ambiguous (missing-inputs could also be an orphan or trying to
    # spend nonexistent coins).
    later_txids = set()
    for tx in txns:
        later_txids.add(tx.get_hash())

    # Package must not contain any duplicate transactions, which is checked by txid. This also
    # includes transactions with duplicate wtxids and same-txid-different-witness transactions.
    if len(later_txids) != len(txns):
        state.invalid(PackageValidationResult.PCKG_POLICY, "package-contains-duplicates")
        return False

    for tx in txns:
        for input in tx.vin:
            if input.prevout.hash in later_txids:
                # The parent is a subsequent transaction in the package.
                state.invalid(PackageValidationResult.PCKG_POLICY, "package-not-sorted")
                return False

        later_txids.remove(tx.get_hash())

    # Don't allow any conflicting transactions, i.e. spending the same inputs, in a package.
    inputs_seen = set()
    for tx in txns:
        for input in tx.vin:
            if input.prevout in inputs_seen:
                # This input is also present in another tx in the package.
                state.invalid(PackageValidationResult.PCKG_POLICY, "conflict-in-package")
                return False

        # Batch-add all the inputs for a tx at a time. If we added them 1 at a time, we could
        # catch duplicate inputs within a single tx.  This is a more severe, consensus error,
        # and we want to report that from CheckTransaction instead.
        inputs_seen.update([input.prevout for input in tx.vin])

    return True

def is_child_with_parents(package: List[Transaction]) -> bool:
    """Checks if the given package is a child with parents tree.

    Args:
        package: The package of transactions.

    Returns:
        True if the package is a child with parents tree, False otherwise.
    """

    assert all(tx is not None for tx in package)
    if len(package) < 2:
        return False

    # The package is expected to be sorted, so the last transaction is the child.
    child = package[-1]
    input_txids = set(input.prevout.hash for input in child.vin)

    # Every transaction must be a parent of the last transaction in the package.
    return all(ptx.get_hash() in input_txids for ptx in package[:-1])

def is_child_with_parents_tree(package: List[Transaction]) -> bool:
    """Checks if the given package is a child with parents tree, where each parent has no inputs
    from the other parents.

    Args:
        package: The package of transactions.
        
    """

    assert all(tx is not None for tx in package)
    if len(package) < 2:
        return False

    # The package is expected to be sorted, so the last transaction is the child.
    child = package[-1]
    input_txids = set(input.prevout.hash for input in child.vin)

    # Every transaction must be a parent of the last transaction in the package, and no parent
    # can have inputs from other parents.
    for ptx in package[:-1]:
        if ptx.get_hash() not in input_txids or any(input.prevout.hash in input_txids for input in ptx.vin):
            return False

    return True

#----------------------------------------------------------------------------------------------------

from collections import defaultdict
from decimal import Decimal
from typing import Dict, List, Optional, Set

from kernel.mempool_entry import CTxMemPoolEntry
from policy.feerate import CFeeRate
from primitives.transaction import CTransaction
from txmempool import CTxMemPool

MAX_REPLACEMENT_CANDIDATES = 100

def is_rbf_opt_in(tx: CTransaction, pool: CTxMemPool) -> RBFTransactionState:
    """Checks if the given transaction is opt-in for RBF.

    Args:
        tx: The transaction.
        pool: The mempool.

    Returns:
        The RBF transaction state.
    """

    with pool.cs:
        # First check the transaction itself.
        if tx.is_rbf_opt_in():
            return RBFTransactionState.REPLACEABLE_BIP125

        # If this transaction is not in our mempool, then we can't be sure
        # we will know about all its inputs.
        if not pool.exists(tx.txid):
            return RBFTransactionState.UNKNOWN

        # If all the inputs have nSequence >= maxint-1, it still might be
        # signaled for RBF if any unconfirmed parents have signaled.
        entry = pool.map_tx[tx.txid]
        ancestors = pool.assume_calculate_mempool_ancestors(
            entry, CTxMemPool.Limits.NoLimits(), False
        )

        for ancestor in ancestors:
            if ancestor.tx.is_rbf_opt_in():
                return RBFTransactionState.REPLACEABLE_BIP125

        return RBFTransactionState.FINAL

def is_rbf_opt_in_empty_mempool(tx: CTransaction) -> RBFTransactionState:
    """Checks if the given transaction is opt-in for RBF in an empty mempool.

    Args:
        tx: The transaction.

    Returns:
        The RBF transaction state.
    """

    # If we don't have a local mempool we can only check the transaction itself.
    return tx.is_rbf_opt_in() or RBFTransactionState.UNKNOWN

def get_entries_for_conflicts(tx: CTransaction,
                              pool: CTxMemPool,
                              iters_conflicting: Set[CTxMemPoolEntry],
                              all_conflicts: Set[CTxMemPoolEntry]) -> Optional[str]:
    """Gets the entries for the conflicts of the given transaction.

    Args:
        tx: The transaction.
        pool: The mempool.
        iters_conflicting: The set of conflicting transactions.
        all_conflicts: The set of all conflicts.

    Returns:
        A string containing the list of conflicting transactions, or None if there are no conflicts.
    """

    with pool.cs:
        txid = tx.txid
        n_conflicting_count = 0
        for entry in iters_conflicting:
            n_conflicting_count += entry.get_count_with_descendants()
            # Rule #5: don't consider replacing more than MAX_REPLACEMENT_CANDIDATES
            # entries from the mempool. This potentially overestimates the number of actual
            # descendants (i.e. if multiple conflicts share a descendant, it will be counted multiple
            # times), but we just want to be conservative to avoid doing too much work.
            if n_conflicting_count > MAX_REPLACEMENT_CANDIDATES:
                return f"rejecting replacement {txid}; too many potential replacements ({n_conflicting_count} > {MAX_REPLACEMENT_CANDIDATES})\n"

        # Calculate the set of all transactions that would have to be evicted.
        for entry in iters_conflicting:
            pool.calculate_descendants(entry, all_conflicts)

        return None

def has_no_new_unconfirmed(tx: CTransaction,
                            pool: CTxMemPool,
                            iters_conflicting: Set[CTxMemPoolEntry]) -> Optional[str]:
    """Checks if the given transaction has no new unconfirmed inputs.

    Args:
        tx: The transaction.
        pool: The mempool.
        iters_conflicting: The set of conflicting transactions.

    Returns:
        A string containing the list of new unconfirmed inputs, or None if there are no new unconfirmed inputs."""
                                  with pool.cs:
        parents_of_conflicts = set()
        for entry in iters_conflicting:
            for txin in entry.tx.vin:
                parents_of_conflicts.add(txin.prevout.hash)

        for j in range(len(tx.vin)):
            # Rule #2: We don't want to accept replacements that require low feerate junk to be
            # mined first.  Ideally we'd keep track of the ancestor feerates and make the decision
            # based on that, but for now requiring all new inputs to be confirmed works.
            #
            # Note that if you relax this to make RBF a little more useful, this may break the
            # CalculateMempoolAncestors RBF relaxation which subtracts the conflict count/size from the
            # descendant limit.
            if tx.vin[j].prevout.hash not in parents_of_conflicts:
                # Rather than check the UTXO set - potentially expensive - it's cheaper to just check
                # if the new input refers to a tx that's in the mempool.
                if pool.exists(tx.vin[j].prevout.hash):
                    return f"replacement {tx.txid} adds unconfirmed input, idx {j}"

        return None
#----------------------------------------------------------------------------------------------------------------

def entries_and_txids_disjoint(ancestors: Set[CTxMemPoolEntry],
                               direct_conflicts: Set[uint256],
                               txid: uint256) -> Optional[str]:
    """Checks if the given transaction does not spend any conflicts.

    Args:
        ancestors: The set of ancestors of the given transaction.
        direct_conflicts: The set of direct conflicts of the given transaction.
        txid: The hash of the given transaction.

    Returns:
        A string containing the list of conflicting transactions, or None if there are no conflicts.
    """

    for ancestor in ancestors:
        hash_ancestor = ancestor.tx.GetHash()
        if direct_conflicts.count(hash_ancestor):
            return f"{txid} spends conflicting transaction {hash_ancestor}"

    return None

def pays_more_than_conflicts(iters_conflicting: Set[CTxMemPoolEntry],
                             replacement_feerate: CFeeRate,
                             txid: uint256) -> Optional[str]:
    """Checks if the given transaction pays more than the conflicting transactions.

    Args:
        iters_conflicting: The set of conflicting transactions.
        replacement_feerate: The feerate of the given transaction.
        txid: The hash of the given transaction.

    Returns:
        A string containing the list of conflicting transactions, or None if the given transaction pays more than the conflicting transactions.
    """

    for entry in iters_conflicting:
        original_feerate = CFeeRate(entry.GetModifiedFee(), entry.GetTxSize())
        if replacement_feerate <= original_feerate:
            return f"rejecting replacement {txid}; new feerate {replacement_feerate} <= old feerate {original_feerate}"

    return None

def pays_for_rbf(original_fees: CAmount,
                 replacement_fees: CAmount,
                 replacement_vsize: size_t,
                 relay_fee: CFeeRate,
                 txid: uint256) -> Optional[str]:
    """Checks if the given transaction pays for itself and the conflicting transactions.

    Args:
        original_fees: The fees of the conflicting transactions.
        replacement_fees: The fees of the given transaction.
        replacement_vsize: The size of the given transaction.
        relay_fee: The relay fee.
        txid: The hash of the given transaction.

    Returns:
        A string containing the list of errors, or None if the given transaction pays for itself and the conflicting transactions.
    """

    if replacement_fees < original_fees:
        return f"rejecting replacement {txid}, less fees than conflicting txs; {FormatMoney(replacement_fees)} < {FormatMoney(original_fees)}"

    additional_fees = replacement_fees - original_fees
    if additional_fees < relay_fee.GetFee(replacement_vsize):
        return f"rejecting replacement {txid}, not enough additional fees to relay; {FormatMoney(additional_fees)} < {FormatMoney(relay_fee.GetFee(replacement_vsize))}"

    return None

#---------------------------------------------------------------------------------------------

# Copyright (c) 2009-2010 Satoshi Nakamoto
# Copyright (c) 2009-2022 The Bitcoin Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.

from typing import Optional

from policy.policymax import DEFAULT_BYTES_PER_SIGOP

# The number of bytes per signature operation.
n_bytes_per_sig_op: Optional[int] = DEFAULT_BYTES_PER_SIGOP



