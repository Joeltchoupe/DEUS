import enum
import math
import typing

from dataclasses import dataclass

from ixogen import (
    consensus,
    hashes,
    serialize,
    script,
    tx,
    util,
)

# Enums for different types of script outputs.
@enum.unique
class TxoutType(enum.Enum):
    STANDARD = 0
    NONSTANDARD = 1
    MULTISIG = 2
    NULL_DATA = 3
    WITNESS_UNKNOWN = 4


@dataclass
class Coin:
    outpoint: tx.OutPoint
    out: tx.TxOut

# A helper class that provides access to a set of coins.
class CoinsViewCache:
    def __init__(self, coins: typing.List[Coin]):
        self._coins = coins

    def AccessCoin(self, outpoint: tx.OutPoint) -> Coin:
        for coin in self._coins:
            if coin.outpoint == outpoint:
                return coin
        raise KeyError(f"Outpoint {outpoint} not found in coins view cache.")

# Returns the dust threshold for a given transaction output, based on the given
# fee rate. Dust is any output that is smaller than the dust threshold.
def GetDustThreshold(txout: tx.TxOut, dust_relay_fee_in: consensus.FeeRate) -> int:
    """
    GetDustThreshold returns the dust threshold for a given transaction output,
    based on the given fee rate. Dust is any output that is smaller than the dust
    threshold.

    Args:
        txout: The transaction output to get the dust threshold for.
        dust_relay_fee_in: The fee rate to use to calculate the dust threshold.

    Returns:
        The dust threshold for the given transaction output.
    """

    # "Dust" is defined in terms of dustRelayFee,
    # which has units satoshis-per-kilobyte.
    # If you'd pay more in fees than the value of the output
    # to spend something, then we consider it dust.
    # A typical spendable non-segwit txout is 34 bytes big, and will
    # need a CTxIn of at least 148 bytes to spend:
    # so dust is a spendable txout less than
    # 182*dustRelayFee/1000 (in satoshis).
    # 546 satoshis at the default rate of 3000 sat/kvB.
    # A typical spendable segwit P2WPKH txout is 31 bytes big, and will
    # need a CTxIn of at least 67 bytes to spend:
    # so dust is a spendable txout less than
    # 98*dustRelayFee/1000 (in satoshis).
    # 294 satoshis at the default rate of 3000 sat/kvB.
    if txout.scriptPubKey.IsUnspendable():
        return 0

    size = serialize.GetSize(txout)
    witness_version = 0
    witness_program = b""

    # Note this computation is for spending a Segwit v0 P2WPKH output (a 33 bytes
    # public key + an ECDSA signature). For Segwit v1 Taproot outputs the minimum
    # satisfaction is lower (a single BIP340 signature) but this computation was
    # kept to not further reduce the dust level.
    # See discussion in https://github.com/bitcoin/bitcoin/pull/22779 for details.
    if txout.scriptPubKey.IsWitnessProgram(witness_version, witness_program):
        # sum the sizes of the parts of a transaction input
        # with 75% segwit discount applied to the script size.
        size += (32 + 4 + 1 + (107 / script.WITNESS_SCALE_FACTOR) + 4)
    else:
        size += (32 + 4 + 1 + 107 + 4)  # the 148 mentioned above

    return dust_relay_fee_in.GetFee(size)

# Returns whether or not a given transaction output is considered dust, based on
# the given fee rate.
def IsDust(txout: tx.TxOut, dust_relay_fee_in: consensus.FeeRate) -> bool:
    """
    IsDust returns whether or not a given transaction output is considered dust,
    based on the given fee rate.

    Args:
        txout: The transaction output to check for dust.
        dust_relay_fee_in: The fee rate to use to calculate the dust threshold.

    Returns:
        True if the transaction output is considered dust, False otherwise.
    """

    return txout.nValue < GetDustThreshold(txout, dust_relay_fee_in)

# Returns whether or not a given script is considered standard.
def IsStandard(script: script.Script, max_datacarrier_bytes: typing.Optional[int] = None, whichType: TxoutType = TxoutType.STANDARD) -> bool:
    """
    IsStandard returns whether or not a given script is considered standard.

    Args:
        script: The script to check for validity.
        max_datacarrier_bytes: The maximum number of bytes allowed for data
            outputs.
        whichType: The type of output that the script is being evaluated for.

    Returns:
        True if the script is considered standard, False otherwise.
    """

    # Nonstandard scripts are always rejected.

    if script.IsUnspendable():
        return False

    # Standard multisig scripts must have at least 2 signers and at most 3 signers.

    if whichType == TxoutType.MULTISIG:
        solutions = script.GetSolutions()
        if len(solutions) < 2 or len(solutions) > 3:
            return False

    # Standard data outputs must not exceed the maximum allowed size.

    if whichType == TxoutType.NULL_DATA:
        if max_datacarrier_bytes is not None and len(script) > max_datacarrier_bytes:
            return False

    return True

# Returns whether or not a given transaction is considered standard.
def IsStandardTx(tx: tx.Tx, max_datacarrier_bytes: typing.Optional[int] = None, permit_bare_multisig: bool = False, dust_relay_fee: consensus.FeeRate = consensus.FeeRate(1000), reason: str = "") -> bool:
    """
    IsStandardTx returns whether or not a given transaction is considered standard.

    Args:
        tx: The transaction to check for validity.
        max_datacarrier_bytes: The maximum number of bytes allowed for data
            outputs.
        permit_bare_multisig: Whether or not bare multisig outputs are allowed.
        dust_relay_fee: The fee rate to use to calculate the dust threshold.
        reason: A string that will be set to the reason why the transaction is
            not considered standard if it is not standard.

    Returns:
        True if the transaction is considered standard, False otherwise.
    """

    # Transaction version must be within the allowed range.

    if tx.nVersion > TX_MAX_STANDARD_VERSION or tx.nVersion < 1:
        reason = "version"
        return False

    # Transaction size must not exceed the maximum allowed size.

    tx_weight = tx.GetTransactionWeight()
    if tx_weight > MAX_STANDARD_TX_WEIGHT:
        reason = "tx-size"
        return False

    # Input scripts must not be too large.

    for txin in tx.vin:
        if len(txin.scriptSig) > MAX_STANDARD_SCRIPTSIG_SIZE:
            reason = "scriptsig-size"
            return False

    # Output scripts must be standard.

    nDataOut = 0
    for txout in tx.vout:
        whichType = TxoutType.STANDARD
        if not IsStandard(txout.scriptPubKey, max_datacarrier_bytes, whichType):
            reason = "scriptpubkey"
            return False

        if whichType == TxoutType.NULL_DATA:
            nDataOut += 1
        elif whichType == TxoutType.MULTISIG and not permit_bare_multisig:
           
        if whichType == TxoutType.MULTISIG and not permit_bare_multisig:
            reason = "bare-multisig"
            return False
        elif IsDust(txout, dust_relay_fee):
            reason = "dust"
            return False

    # Only one OP_RETURN output is permitted.

    if nDataOut > 1:
        reason = "multi-op-return"
        return False

    return True

# Returns whether or not a given transaction input is considered standard.
def AreInputsStandard(tx: tx.Tx, coins: typing.List[Coin]) -> bool:
    """
    AreInputsStandard returns whether or not a given transaction input is
    considered standard.

    Args:
        tx: The transaction to check the inputs for.
        coins: A list of coins that the transaction is spending.

    Returns:
        True if the transaction input is considered standard, False otherwise.
    """

    # Coinbase transactions don't use inputs normally.

    if tx.IsCoinBase():
        return True

    for txin in tx.vin:
        # The previous output must be standard.
        coin = coins.get(txin.prevout)
        if coin is None:
            return False

        if not IsStandard(coin.out.scriptPubKey):
            return False

        # P2SH inputs must not have too many signature operations.
        if txin.scriptSig.IsPushOnly() is False:
            return False

        if txin.scriptSig.GetSigOpCount(True) > MAX_P2SH_SIGOPS:
            return False

    return True

#------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List

from ixogen import (
    consensus,
    script,
    tx,
    util,
)

# Enums for different types of script outputs.
@enum.unique
class TxoutType(enum.Enum):
    STANDARD = 0
    NONSTANDARD = 1
    MULTISIG = 2
    NULL_DATA = 3
    WITNESS_UNKNOWN = 4

# A helper class that provides access to a set of coins.
class CoinsViewCache:
    def __init__(self, coins: List[Coin]):
        self._coins = coins

    def AccessCoin(self, outpoint: tx.OutPoint) -> Coin:
        for coin in self._coins:
            if coin.outpoint == outpoint:
                return coin
        raise KeyError(f"Outpoint {outpoint} not found in coins view cache.")

# Returns whether or not a given witness program is considered standard.
def IsWitnessStandard(scriptPubKey: script.Script, txin: tx.TxIn, coins: CoinsViewCache) -> bool:
    """
    IsWitnessStandard returns whether or not a given witness program is considered standard.

    Args:
        scriptPubKey: The scriptPubKey to check.
        txin: The txin to check.
        coins: A coins view cache to access the previous outputs.

    Returns:
        True if the witness program is considered standard, False otherwise.
    """

    witnessVersion = 0
    witnessProgram = b""

    # Non-witness programs must not be associated with any witness.
    if not scriptPubKey.IsWitnessProgram(witnessVersion, witnessProgram):
        return False

    prev = coins.AccessCoin(txin.prevout).out

    # Check P2WSH standard limits.
    if witnessVersion == 0 and len(witnessProgram) == script.WITNESS_V0_SCRIPTHASH_SIZE:
        if len(txin.scriptWitness.stack.back()) > script.MAX_STANDARD_P2WSH_SCRIPT_SIZE:
            return False
        sizeWitnessStack = len(txin.scriptWitness.stack) - 1
        if sizeWitnessStack > script.MAX_STANDARD_P2WSH_STACK_ITEMS:
            return False
        for stackItem in txin.scriptWitness.stack[:-1]:
            if len(stackItem) > script.MAX_STANDARD_P2WSH_STACK_ITEM_SIZE:
                return False

    # Check policy limits for Taproot spends:
    # - MAX_STANDARD_TAPSCRIPT_STACK_ITEM_SIZE limit for stack item size
    # - No annexes
    if witnessVersion == 1 and len(witnessProgram) == script.WITNESS_V1_TAPROOT_SIZE and not scriptPubKey.IsPayToScriptHash():
        # Taproot spend (non-P2SH-wrapped, version 1, witness program size 32; see BIP 341)
        if len(txin.scriptWitness.stack) >= 2 and txin.scriptWitness.stack[-1] and txin.scriptWitness.stack[-1][0] == script.ANNEX_TAG:
            # Annexes are nonstandard as long as no semantics are defined for them.
            return False
        if len(txin.scriptWitness.stack) >= 2:
            # Script path spend (2 or more stack elements after removing optional annex)
            controlBlock = txin.scriptWitness.stack.pop()
            txin.scriptWitness.stack.pop()  # Ignore script
            if not controlBlock:
                return False  # Empty control block is invalid
            if (controlBlock[0] & script.TAPROOT_LEAF_MASK) == script.TAPROOT_LEAF_TAPSCRIPT:
                # Leaf version 0xc0 (aka Tapscript, see BIP 342)
                for stackItem in txin.scriptWitness.stack:
                    if len(stackItem) > script.MAX_STANDARD_TAPSCRIPT_STACK_ITEM_SIZE:
                        return False

    return True

# Returns the virtual transaction size in bytes.
def GetVirtualTransactionSize(nWeight: int64_t, nSigOpCost: int64_t, bytes_per_sigop: unsigned int) -> int64_t:
    """
    GetVirtualTransactionSize returns the virtual transaction size in bytes.

    Args:
        nWeight: The transaction weight.
        nSigOpCost: The number of signature operations in the transaction.
        bytes_per_sigop: The number of bytes per signature operation.

    Returns:
        The virtual transaction size in bytes.
    """

    return (max(nWeight, nSigOpCost * bytes_per_sigop) + WITNESS_SCALE_FACTOR - 1) // WITNESS_SCALE_FACTOR

# Returns the virtual transaction size of a given transaction.
def GetVirtualTransactionSize(tx: tx.Tx, nSigOpCost: int64_t, bytes_per_sigop: unsigned int) -> int64_t:
    """
    GetVirtualTransactionSize returns the virtual transaction size of a given transaction.

    Args:
        tx: The transaction to get the virtual size of.
        nSigOpCost: The number of signature operations in the transaction.
        bytes_per_sigop: The number of bytes per signature operation.

    Returns:
        The virtual transaction size in bytes.
    """

    return GetVirtualTransactionSize(tx.GetTransactionWeight(), nSigOpCost, bytes_per_sigop)

# Returns the virtual transaction size of a given transaction input.
def GetVirtualTransactionInputSize(txin: tx.TxIn, nSigOpCost: int64_t, bytes_per_sigop: unsigned int) -> int64_t:
    """
    GetVirtualTransactionInputSize returns the virtual transaction size of a given transaction input.

    Args:
        txin: The transaction input to get the virtual size of.
        nSigOpCost: The number of signature operations in the input.
        bytes_per_sigop: The number of bytes per signature operation.

    Returns:
        The virtual transaction size in bytes.
    """

    return GetVirtualTransactionSize(txin.GetTransactionInputWeight(), nSigOpCost, bytes_per_sigop)
#------------------------------------------------------------------------------------------------------------------------------------------------

def IsWitnessStandard(scriptPubKey: script.Script, txin: tx.TxIn, coins: CoinsViewCache) -> bool:
    """
    IsWitnessStandard returns whether or not a given witness program is considered standard.

    Args:
        scriptPubKey: The scriptPubKey to check.
        txin: The txin to check.
        coins: A coins view cache to access the previous outputs.

    Returns:
        True if the witness program is considered standard, False otherwise.
    """

    witnessVersion = 0
    witnessProgram = b""

    # Non-witness programs must not be associated with any witness.
    if not scriptPubKey.IsWitnessProgram(witnessVersion, witnessProgram):
        return False

    prev = coins.AccessCoin(txin.prevout).out

    # Check P2WSH standard limits.
    if witnessVersion == 0 and len(witnessProgram) == script.WITNESS_V0_SCRIPTHASH_SIZE:
        if len(txin.scriptWitness.stack.back()) > script.MAX_STANDARD_P2WSH_SCRIPT_SIZE:
            return False
        sizeWitnessStack = len(txin.scriptWitness.stack) - 1
        if sizeWitnessStack > script.MAX_STANDARD_P2WSH_STACK_ITEMS:
            return False
        for stackItem in txin.scriptWitness.stack[:-1]:
            if len(stackItem) > script.MAX_STANDARD_P2WSH_STACK_ITEM_SIZE:
                return False

    # Check policy limits for Taproot spends:
    # - MAX_STANDARD_TAPSCRIPT_STACK_ITEM_SIZE limit for stack item size
    # - No annexes
    if witnessVersion == 1 and len(


