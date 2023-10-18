#the mempool of the bpsc is a volatile space in the RAM of 10 seconds' expiry deadline, where hte bpsc stores the transactions received in order to process it; 
#after 10 seconds it should have validated and created a block of these transactions, otherwise they are directly suppressed or redirected to another bpsc

//les frais de transactions collectés par le bpsc sont envoyés au wallet father et il la coinbase est envoyée au noeud du bpsc et utilisable après 
// Vérifie si une transaction est valide et peut être ajoutée au mempool

bool AcceptTransaction(const CTransaction& tx, CValidationState& state)
{
  // Vérifie que les scripts d'entrée sont valides
  if (!CheckInputs(tx, state, mempool.mapCoins, SCRIPT_VERIFY_P2SH)) {
    return false;
  }

  // Vérifie que la transaction n'est pas double dépensée
  if (mempool.HasCoinbaseOrSpent(tx.GetHash())) {
    state.Invalid(TxValidationResult::TX_REUSE_OF_OUTPUTS);
    return false;
  }

  // Vérifie que la transaction n'est pas soumise trop tôt
  if (tx.nLockTime > chainActive.Tip()->nTime + LOCKTIME_THRESHOLD) {
    state.Invalid(TxValidationResult::TX_TOO_EARLY);
    return false;
  }

  // Vérifie que la transaction n'est pas soumise avec une signature RBF
  if (tx.HasRBF()) {
    state.Invalid(TxValidationResult::TX_RBF);
    return false;
  }

  // La transaction est valide et peut être ajoutée au mempool
  return true;
}
//----------------------------------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

struct Transaction {
  uint64_t amount;
  string from_address;
  string to_address;
  vector<uint8_t> signature;
};

bool verify_signature(const Transaction& transaction) {
  // Vérifie que la signature de la transaction est valide.
  // ...
}

bool verify_amount(const Transaction& transaction) {
  // Vérifie que le montant de la transaction est valide.
  // ...
}

bool verify_fees(const Transaction& transaction) {
  // Vérifie que les frais de la transaction sont suffisants.
  // ...
}

bool verify_double_spend(const Transaction& transaction, const vector<Transaction>& mempool) {
  // Vérifie que la transaction ne tente pas de dépenser les mêmes bitcoins deux fois.
  // ...
}

int main() {
  // Chargement des transactions entrantes.
  vector<Transaction> transactions;
  for (string line; getline(cin, line);) {
    Transaction transaction;
    istringstream iss(line);
    iss >> transaction.amount >> transaction.from_address >> transaction.to_address >> transaction.signature;
    transactions.push_back(transaction);
  }

  // Vérification des transactions.
  for (Transaction& transaction : transactions) {
    if (!verify_signature(transaction)) {
      cout << "Transaction rejetée : signature invalide" << endl;
      continue;
    }
    if (!verify_amount(transaction)) {
      cout << "Transaction rejetée : montant invalide" << endl;
      continue;
    }
    if (!verify_fees(transaction)) {
      cout << "Transaction rejetée : frais insuffisants" << endl;
      continue;
    }
    if (!verify_double_spend(transaction, transactions)) {
      cout << "Transaction rejetée : double dépense" << endl;
      continue;
    }

    // La transaction est valide.
    cout << "Transaction acceptée" << endl;
  }

  return 0;
}

//---------------------------------------------------------------------------------------------------

//Le fichier mempool carrier est un fichier Python qui est utilisé par les nœuds Bitcoin pour transporter les transactions de la mempool d'un nœud à un autre. La mempool est un ensemble de transactions qui n'ont pas encore été incluses dans un bloc.

//Le fichier mempool carrier implémente un protocole simple pour le transport des transactions. Ce protocole utilise un format de données binaire pour représenter les transactions.

//Le fichier mempool carrier est un élément important de la communication entre les nœuds Bitcoin. Il permet aux nœuds de partager les transactions entre eux, ce qui est nécessaire pour maintenir la cohérence de la blockchain.
#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

struct Transaction {
  uint64_t amount;
  string from_address;
  string to_address;
  vector<uint8_t> signature;
};

class MempoolCarrier {
 public:
  static vector<uint8_t> encode_transactions(const vector<Transaction>& transactions) {
    // Encodage des transactions.
    vector<uint8_t> encoded_transactions;
    for (const Transaction& transaction : transactions) {
      encoded_transactions.insert(encoded_transactions.end(), transaction.amount);
      encoded_transactions.insert(encoded_transactions.end(), transaction.from_address.begin(), transaction.from_address.end());
      encoded_transactions.insert(encoded_transactions.end(), transaction.to_address.begin(), transaction.to_address.end());
      encoded_transactions.insert(encoded_transactions.end(), transaction.signature.begin(), transaction.signature.end());
    }
    return encoded_transactions;
  }

  static vector<Transaction> decode_transactions(const vector<uint8_t>& encoded_transactions) {
    // Décodage des transactions.
    vector<Transaction> transactions;
    for (int i = 0; i < encoded_transactions.size(); i += sizeof(Transaction)) {
      Transaction transaction;
      transaction.amount = encoded_transactions[i];
      memcpy(transaction.from_address.data(), encoded_transactions.data() + i + sizeof(transaction.amount), sizeof(transaction.from_address));
      memcpy(transaction.to_address.data(), encoded_transactions.data() + i + sizeof(transaction.amount) + sizeof(transaction.from_address), sizeof(transaction.to_address));
      memcpy(transaction.signature.data(), encoded_transactions.data() + i + sizeof(transaction.amount) + sizeof(transaction.from_address) + sizeof(transaction.to_address), sizeof(transaction.signature));
      transactions.push_back(transaction);
    }
    return transactions;
  }
};

int main() {
  // Chargement des transactions.
  vector<Transaction> transactions;
  for (string line; getline(cin, line);) {
    Transaction transaction;
    istringstream iss(line);
    iss >> transaction.amount >> transaction.from_address >> transaction.to_address >> transaction.signature;
    transactions.push_back(transaction);
  }

  // Encodage des transactions.
  vector<uint8_t> encoded_transactions = MempoolCarrier::encode_transactions(transactions);

  // Transmission des transactions.
  // ...

  // Réception des transactions.
  // ...

  // Décodage des transactions.
  vector<Transaction> received_transactions = MempoolCarrier::decode_transactions(received_data);

  // Ajout des transactions à la mempool.
  // ...

  return 0;
}

//-------------------------------------------------------------------------------------
oid remove_expired_transactions() {
    // Supprime les transactions expirées.
    for (int i = 0; i < transactions.size();) {
      if (transactions[i].expiry_time < time(NULL)) {
        transactions.erase(transactions.begin() + i);
      } else {
        i++;
      }
    }
  }

 private:
  vector<Transaction> transactions;

  bool is_valid_transaction(const Transaction& transaction) {
    // Vérifie que la transaction est valide.
    // ...

    return true;
  }
};

int main() {
  // Création d'un mempool.
  Mempool mempool;

  // Ajout d'une transaction valide.
  Transaction transaction;
  transaction.amount = 1000;
  transaction.from_address = "1234567890";
  transaction.to_address = "9876543210";
  mempool.add_transaction(transaction);

  // Ajout d'une transaction avec un dust output.
  transaction.amount = 1;
  mempool.add_transaction(transaction);

  // Vérification du mempool.
  cout << "Nombre de transactions : " << mempool.transactions.size() << endl;

  // Expiration des transactions.
  mempool.remove_expired_transactions();

  // Vérification du mempool après expiration.
  cout << "Nombre de transactions : " << mempool.transactions.size() << endl;

  return 0;
}













































// Classe BPSCMiner
class BPSCMiner : public IMiner
{
public:
  // Constructeur
  BPSCMiner() {}

  // Méthode pour miner un bloc
  Block* MineBlock(const std::vector<Transaction>& transactions) override
  {
    // ...

    // Utiliser le BPSC pour valider les transactions
    bpsc->ValidateTransactions(transactions);

    // ...

    // Créer la coinbase
    Block* block = new Block();
    block->vtx.push_back(CreateCoinbaseTransaction(bpsc->GetPublicKey()));

    // ...

    return block;
  }

private:
  // Objet BPSC
  BPSC* bpsc;
};

// Fonction pour créer un BPSCMiner
IMiner* CreateMiner()
{
  // Créer un BPSC
  BPSC* bpsc = new BPSC();

  // Créer un BPSCMiner
  BPSCMiner* miner = new BPSCMiner();
  miner->bpsc = bpsc;

  return miner;
}

// Fonction pour ajouter une transaction
void AddTransaction(IBlockchain* blockchain, const Transaction& transaction)
{
  // ...

  // Vérifier que la transaction est valide
  if (!bpsc->ValidateTransaction(transaction))
  {
    // ...
  }

  // ...
}

// Fonction pour créer une coinbase transaction
Transaction* CreateCoinbaseTransaction(const std::string& public_key)
{
  // ...

  // Diviser la coinbase en trois parts
  Transaction* coinbase = new Transaction();
  coinbase->vin.push_back(CTxIn(COutPoint(0, 0), 0));
  coinbase->vout.push_back(CTxOut(25, CScript(public_key)));
  coinbase->vout.push_back(CTxOut(25, CScript(public_key)));
  coinbase->vout.push_back(CTxOut(25, CScript(GetFullNodesPublicKey())));

  return coinbase;
}

// Fonction pour obtenir la récompense du bloc
int64_t GetBlockReward()
{
  // ...
#----------------------------------------------------------------------------------------------------------------------------
Les utilisateurs envoient des transactions au contrat intelligent. Chaque utilisateur envoie sa tx au bpsc le plus proche dans le réseau
Le contrat intelligent collecte les transactions et les ajoute à une file d'attente.
Le contrat intelligent  a ensuite la possibilité de créer un nouveau bloc à partir des transactions de la file d'attente.
Le contrat intelligent valide le nouveau bloc et l'ajoute à la blockchain si le nombre de signatures des autres bpsc à été atteint.
