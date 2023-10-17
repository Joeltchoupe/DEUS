from consensus import amount
from primitives import transaction
from consensus import validation


def check_transaction(tx, state):
    """
    Vérifie la validité de la transaction.

    Args:
        tx: La transaction à vérifier.
        state: L'état de validation de la transaction.

    Returns:
        True si la transaction est valide, False sinon.
    """

    # Vérifications de base qui ne dépendent d'aucun contexte
    if not tx.vin:
        return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-txns-vin-empty")
    if not tx.vout:
        return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-txns-vout-empty")

    # Limites de taille (cela ne prend pas en compte le témoin, car il n'a pas encore été vérifié pour la malléabilité)
    if (
        len(tx.serialize(PROTOCOL_VERSION | SERIALIZE_TRANSACTION_NO_WITNESS))
        * WITNESS_SCALE_FACTOR
        > MAX_BLOCK_WEIGHT
    ):
        return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-txns-oversize")

    # Vérifie les valeurs de sortie négatives ou débordantes (voir CVE-2010-5139)
    n_value_out = 0
    for txout in tx.vout:
        if txout.n_value < 0:
            return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-txns-vout-negative")
        if txout.n_value > MAX_MONEY:
            return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-txns-vout-toolarge")
        n_value_out += txout.n_value
        if not MoneyRange(n_value_out):
            return state.invalid(
                TxValidationResult.TX_CONSENSUS, "bad-txns-txouttotal-toolarge"
            )

    # Vérifie les entrées dupliquées (voir CVE-2018-17144)
    # Bien que Consensus.CheckTxInputs vérifie si toutes les entrées d'une transaction sont disponibles, et que UpdateCoins marque toutes les entrées
    # d'une transaction comme dépensées, il ne vérifie pas si la transaction a des entrées dupliquées.
    # Si cette vérification n'est pas effectuée, cela entraînera soit un crash, soit un bug d'inflation, en fonction de l'implémentation de la base de données de pièces sous-jacente.
    v_in_out_points = set()
    for txin in tx.vin:
        if txin.prevout in v_in_out_points:
            return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-txns-inputs-duplicate")
        v_in_out_points.add(txin.prevout)

    if tx.is_coinbase():
        if not (2 <= len(tx.vin[0].script_sig) <= 100):
            return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-cb-length")
    else:
        for txin in tx.vin:
            if txin.prevout.is_null():
                return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-txns-prevout-null")

    return True

