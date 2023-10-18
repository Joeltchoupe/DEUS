#TX_CHECK
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

#TX_VALIDATION----------------------------------------------------------------------------------------------------------------------------------------------------
import chain
import coins
from consensus import amount
from consensus import consensus
from consensus import validation
from primitives import transaction
from script import interpreter
from util import check
from util import moneystr

def consensus_check_tx_inputs(tx, state, inputs, n_spend_height, txfee):
    """
    Vérifie que les entrées de la transaction sont disponibles et valides.

    Args:
        tx: La transaction à vérifier.
        state: L'état de validation de la transaction.
        inputs: Un cache de vue des pièces de monnaie.
        n_spend_height: La hauteur du bloc dans lequel la transaction est dépensée.
        txfee: Les frais de transaction.

    Returns:
        True si les entrées de la transaction sont disponibles et valides, False sinon.
    """

    # Les entrées de la transaction sont-elles disponibles ?
    if not inputs.have_inputs(tx):
        return state.invalid(TxValidationResult.TX_MISSING_INPUTS, "bad-txns-inputs-missingorspent",
                             strprintf("%s: inputs missing/spent", __func__))

    n_value_in = 0
    for i in range(len(tx.vin)):
        prevout = tx.vin[i].prevout
        coin = inputs.access_coin(prevout)
        assert not coin.is_spent()

        # Si prev est un coinbase, vérifiez qu'il est mature
        if coin.is_coinbase() and n_spend_height - coin.n_height < COINBASE_MATURITY:
            return state.invalid(TxValidationResult.TX_PREMATURE_SPEND, "bad-txns-premature-spend-of-coinbase",
                                 strprintf("tried to spend coinbase at depth %d", n_spend_height - coin.n_height))

        # Vérifiez les valeurs d'entrée négatives ou débordantes
        n_value_in += coin.out.n_value
        if not MoneyRange(coin.out.n_value) or not MoneyRange(n_value_in):
            return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-txns-inputvalues-outofrange")

    value_out = tx.get_value_out()
    if n_value_in < value_out:
        return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-txns-in-belowout",
                             strprintf("value in (%s) < value out (%s)", FormatMoney(n_value_in), FormatMoney(value_out)))

    # Calcul des frais de transaction
    txfee_aux = n_value_in - value_out
    if not MoneyRange(txfee_aux):
        return state.invalid(TxValidationResult.TX_CONSENSUS, "bad-txns-fee-outofrange")

    txfee = txfee_aux
    return True

