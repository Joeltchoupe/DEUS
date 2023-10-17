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

