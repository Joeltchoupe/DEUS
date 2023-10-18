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
