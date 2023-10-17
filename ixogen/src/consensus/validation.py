from .consensus import validation
from .primitives import block
from .primitives import transaction
from .consensus import consensus


class TxValidationState(validation.ValidationState[validation.TxValidationResult]):
    pass


class BlockValidationState(validation.ValidationState[validation.BlockValidationResult]):
    pass


def get_transaction_weight(tx: transaction.CTransaction) -> int32:
    return (
        validation.GetSerializeSize(tx, validation.PROTOCOL_VERSION | validation.SERIALIZE_TRANSACTION_NO_WITNESS)
        * (validation.WITNESS_SCALE_FACTOR - 1)
        + validation.GetSerializeSize(tx, validation.PROTOCOL_VERSION)
    )


def get_block_weight(block: block.CBlock) -> int64:
    return (
        validation.GetSerializeSize(block, validation.PROTOCOL_VERSION | validation.SERIALIZE_TRANSACTION_NO_WITNESS)
        * (validation.WITNESS_SCALE_FACTOR - 1)
        + validation.GetSerializeSize(block, validation.PROTOCOL_VERSION)
    )


def get_transaction_input_weight(txin: transaction.CTxIn) -> int64:
    return (
        validation.GetSerializeSize(txin, validation.PROTOCOL_VERSION | validation.SERIALIZE_TRANSACTION_NO_WITNESS)
        * (validation.WITNESS_SCALE_FACTOR - 1)
        + validation.GetSerializeSize(txin, validation.PROTOCOL_VERSION)
        + validation.GetSerializeSize(txin.scriptWitness.stack, validation.PROTOCOL_VERSION)
    )


def get_witness_commitment_index(block: block.CBlock) -> int:
    commitpos = consensus.NO_WITNESS_COMMITMENT
    if not block.vtx:
        return commitpos

    for o in range(len(block.vtx[0].vout)):
        vout = block.vtx[0].vout[o]
        if (
            len(vout.scriptPubKey) >= consensus.MINIMUM_WITNESS_COMMITMENT
            and vout.scriptPubKey[0] == consensus.OP_RETURN
            and vout.scriptPubKey[1] == 0x24
            and vout.scriptPubKey[2] == 0xaa
            and vout.scriptPubKey[3] == 0x21
            and vout.scriptPubKey[4] == 0xa9
            and vout.scriptPubKey[5] == 0xed
        ):
            commitpos = o
            break

    return commitpos

