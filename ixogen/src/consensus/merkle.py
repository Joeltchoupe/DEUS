import hashlib

def ComputeMerkleRoot(hashes, mutated=False):
  """Calcule le Merkle Root d'une liste de hachages.

  Args:
    hashes: Une liste de hachages.
    mutated: Un booléen indiquant si les hachages ont été modifiés.

  Returns:
    Le Merkle Root des hachages.
  """

  mutation = False
  while len(hashes) > 1:
    if mutated:
      for pos in range(0, len(hashes) - 1, 2):
        if hashes[pos] == hashes[pos + 1]:
          mutation = True
    if len(hashes) % 2 == 1:
      hashes.append(hashes[-1])
    SHA256D64(hashes[0].begin(), hashes[0].begin(), len(hashes) // 2)
    hashes.resize(len(hashes) // 2)
  if mutated:
    mutated = mutation
  if len(hashes) == 0:
    return hashlib.sha256().hexdigest()
  return hashes[0]


def BlockMerkleRoot(block, mutated=False):
  """Calcule le Merkle Root d'un bloc Ixogen.

  Args:
    block: Un objet CBlock.
    mutated: Un booléen indiquant si les hachages ont été modifiés.

  Returns:
    Le Merkle Root du bloc.
  """

  leaves = []
  for tx in block.vtx:
    leaves.append(tx.GetHash())
  return ComputeMerkleRoot(leaves, mutated)


def BlockWitnessMerkleRoot(block, mutated=False):
  """Calcule le Merkle Root des témoins d'un bloc Ixogen.

  Args:
    block: Un objet CBlock.
    mutated: Un booléen indiquant si les hachages ont été modifiés.

  Returns:
    Le Merkle Root des témoins du bloc.
  """

  leaves = []
  leaves.append(hashlib.sha256().hexdigest())
  for tx in block.vtx[1:]:
    leaves.append(tx.GetWitnessHash())
  return ComputeMerkleRoot(leaves, mutated)
def MerkleProof(hash, hashes, index):
  """Calcule une preuve Merkle pour un hachage donné.

  Args:
    hash: Le hachage pour lequel la preuve est calculée.
    hashes: Une liste de hachages.
    index: L'index du hachage dans la liste.

  Returns:
    Une preuve Merkle.
  """

  proof = []
  while index >= len(hashes) // 2:
    proof.append(hashes[index // 2])
    index = index // 2
  proof.append(hashes[index])
  return proof[::-1]


def VerifyMerkleProof(hash, proof, hashes):
  """Vérifie une preuve Merkle pour un hachage donné.

  Args:
    hash: Le hachage pour lequel la preuve est vérifiée.
    proof: Une preuve Merkle.
    hashes: Une liste de hachages.

  Returns:
    True si la preuve est valide, False sinon.
  """

  for i in range(len(proof) - 1, -1, -1):
    hash = hashlib.sha256(hash.encode("utf-8")).hexdigest()
    if i % 2 == 0:
      hash = hash + proof[i]
    else:
      hash = hash + hashlib.sha256(proof[i].encode("utf-8")).hexdigest()
  return hash == hashes[i // 2]
def MerklePath(hash, hashes):
  """Calcule le chemin Merkle pour un hachage donné.

  Args:
    hash: Le hachage pour lequel le chemin est calculé.
    hashes: Une liste de hachages.

  Returns:
    Un chemin Merkle.
  """

  path = []
  while index >= len(hashes) // 2:
    path.append(hashes[index // 2])
    index = index // 2
  path.append(hashes[index])
  return path[::-1]


def VerifyMerklePath(hash, path, hashes):
  """Vérifie un chemin Merkle pour un hachage donné.

  Args:
    hash: Le hachage pour lequel le chemin est vérifié.
    path: Un chemin Merkle.
    hashes: Une liste de hachages.

  Returns:
    True si le chemin est valide, False sinon.
  """

  for i in range(len(path) - 1, -1, -1):
    hash = hashlib.sha256(hash.encode("utf-8")).hexdigest()
    if i % 2 == 0:
      hash = hash + path[i]
    else:
      hash = hash + hashlib.sha256(path[i].encode("utf-8")).hexdigest()
  return hash == hashes[i // 2]
