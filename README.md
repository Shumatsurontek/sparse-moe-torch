# Sparse MoE Torch

Ce projet implémente un Transformer minimaliste avec une couche Mixture-of-Experts (MoE) sparse, en PyTorch.

## Fonctionnement général

- **SparseMoE** est une couche qui route chaque token d'entrée vers un sous-ensemble d'experts (MLP) selon un score de routage appris.
- Pour chaque token, seuls les `top_k` experts (parmi `num_experts`) sont sélectionnés, ce qui rend l'inférence plus efficace et permet une spécialisation des experts.
- Le routage est effectué par une projection linéaire (`self.router`) qui produit des logits pour chaque expert. On sélectionne les `top_k` plus forts pour chaque token.
- Les sorties des experts sont pondérées par un softmax appliqué uniquement sur les logits top-k, puis sommées pour chaque token.
- Le module est intégré dans un bloc Transformer classique (self-attention + MoE).

## 🧮 Formulation mathématique

Le routage sparse MoE s’écrit ainsi :

- Soit \( E = \{f_1, ..., f_N\} \) les experts, \( N \) leur nombre.
- Pour chaque token \( x \), le routeur calcule des scores :
  \[
  s = Wx + b \in \mathbb{R}^N
  \]
- On sélectionne les indices des \( k \) plus grands scores :
  \[
  I = \text{TopK}(s, k)
  \]
- Les poids de gating sont :
  \[
  g = \text{softmax}(s_I)
  \]
- La sortie du MoE pour ce token est :
  \[
  y = \sum_{i \in I} g_i \cdot f_i(x)
  \]

## Structure du code

- `SparseMoE` : la couche MoE sparse, avec routage dynamique et logs détaillés.
- `TransformerBlockWithMoE` : un bloc Transformer avec attention et MoE.
- `MiniMoETransformer` : un mini-transformer empilant plusieurs blocs.
- Un exemple d'exécution en fin de fichier montre la forme des sorties et la distribution des tokens par expert.

## Logging et interprétation

Le projet utilise le module `logging` pour fournir des informations détaillées sur le routage :
- **Input shape** : forme du batch traité.
- **Routing logits shape** : forme des scores de routage.
- **Top-k indices/values** : indices et valeurs des experts sélectionnés pour chaque token.
- **Top-k gates** : poids softmax appliqués aux sorties des experts sélectionnés.
- **Expert X: tokens processed at k=Y** : nombre de tokens traités par chaque expert à chaque passage top-k.
- **Token count per expert** : total de tokens traités par chaque expert sur tout le batch.

Exemple de log :
```
INFO:SparseMoE:Input shape: torch.Size([2, 10, 128]), num_tokens: 20
INFO:SparseMoE:Routing logits shape: torch.Size([20, 4])
INFO:SparseMoE:Top-k indices: tensor([[1, 3], ...])
INFO:SparseMoE:Top-k values: tensor([[ 0.5933,  0.2351], ...])
INFO:SparseMoE:Top-k gates (softmax over top-k): tensor([[0.5886, 0.4114], ...])
INFO:SparseMoE:Expert 0: tokens processed at k=0: 2
INFO:SparseMoE:Expert 1: tokens processed at k=0: 5
...
INFO:SparseMoE:Token count per expert: tensor([ 6,  8, 16, 10], dtype=torch.int32)
```

Cela permet de vérifier la répartition des tokens et le bon fonctionnement du routage sparse.

## Utilisation

- Installez les dépendances (PyTorch).
- Lancez `python main.py` pour exécuter l'exemple et observer les logs.
- Modifiez les paramètres (nombre d'experts, top_k, dimensions) dans `main.py` pour explorer différents comportements.

## Pour aller plus loin

- Vous pouvez intégrer la couche `SparseMoE` dans vos propres architectures.
- Pour une version compatible Hugging Face Trainer (héritant de `transformers.modeling_utils.PreTrainedModel`), voir la suite.
