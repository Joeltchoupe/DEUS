# Copyright (c) 2009-2010 Satoshi Nakamoto
# Copyright (c) 2009-2021 The Ixogen Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.

class CAmount:
    """ Montant en satoshis (peut être négatif) """

    def __init__(self, value: int):
        self.value = value

    @staticmethod
    def from_ixo(ixo: float) -> CAmount:
        """ Convertit un montant en IXO en satoshis """
        return CAmount(ixo * 100000000)

    @staticmethod
    def to_ixo(satoshis: CAmount) -> float:
        """ Convertit un montant en satoshis en IXO """
        return satoshis.value / 100000000

    def __repr__(self) -> str:
        return f"CAmount({self.value})"

# Le nombre de satoshis dans un IXO
COIN = 100000000

# Le montant maximum de satoshis qui existe (en satoshis)
MAX_MONEY = 100000000000 * COIN

def MoneyRange(nValue: CAmount) -> bool:
    """ Retourne True si le montant est valide, False sinon """
    return (nValue >= 0 and nValue <= MAX_MONEY)

