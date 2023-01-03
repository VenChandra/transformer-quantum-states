# Transformer Quantum States and Interpretability  
Transformer decoder for learning and analyzing ground states of spin chains  

Source code for upcoming paper "Transformer Quantum States and Interpretability" with @durrcommasteven 

(based on work done with @durrcommasteven https://github.com/durrcommasteven/transformer_wavefunctions)

## Transformers for long-range quantum spin chains 

Approximating the ground states of quantum systems with long-range (i.e. volume law) entanglement is an 
important and difficult problem. Conventional techniques such as exact diagonalization and DMRG work well for small system sizes and area law entanglement, respectively, but don't perform well outside of those domains. 

Transformers on the other hand are extremely effective in learning long-range dependencies, making them ideally suited for this task. In our upcoming paper, we autoregressively train a decoder-only transformer using [variational Monte-Carlo](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023358) to find the ground state of an area law system (the critical point transverse field Ising model) as well as a volume law system (the modified Haldane-Shastry model). 

We find that transformers achieve state of the art results, and they does so without any constraints/symmetries enforced by hand to improve training, unlike [this RBM benchmark](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.021021). We compute correlation functions and Renyi entropies in the transformer ground state as accuracy metrics.  

Prior applications of neural networks to spin chains typically consider the vocabulary $0, 1$ (spin down/up). But transformers can handle arbitrary dictionary sizes so we also consider training with arbitrary fixed-size bit strings which combine to give the full sequence of spins. In the paper we compare results across different dictionary sizes.   

We also use [relative positional encodings](https://arxiv.org/abs/1803.02155) in order to obtain a richer expressive power for position-dependent correlations.    

## Saliency metrics 

How does the transformer actually find the ground state of a given system? This question of interpretability is obviously a difficult one, but we make an initial attempt at gaining some insight via saliency metrics. Part of our goal is to see how well saliency metrics meant for language models perform when applied to physical systems. The metrics below seem to demonstrate that the transformer learns the locality/non-locality inherent in the spin chain on its way to the ground state.  

We use feed-forward propagation (comparing logits before and after a spin flip), input x gradient, and a version of integrated attention x gradient. [Input x gradient](https://jalammar.github.io/illustrated-transformer/) is simply $$\lVert X_j \odot \nabla_{X_j}\ell_i \lVert_2,$$ where $X_{j}$ is the embedding vector of the jth token, while $\ell_i$ are the logits at the ith position. 

Our definition of attention x gradient is a combination of those in [this paper](https://arxiv.org/abs/2204.11073) and [this paper](https://arxiv.org/pdf/2004.11207.pdf). Given attention weights $A$ of the lth layer, head h, and positions i,j, it is $$\frac{1}{LH}\sum_{l = 1}^L \sum_{h = 1}^H A^{lh}_{ij}\odot \int_0^1 d\alpha~\text{ReLU}(\nabla_A \ell_i(\alpha A)),$$

## Contents 

This repository contains the following modules: 
* [relativemultiheadattention](relativemultiheadattention.py): implementation of multi-head attention layer with relative positional encodings that also allows one to compute gradients with respect to attention weights using TensorFlow's gradient tape   
* [TransformerRelativeWF](TransformerRelativeWF.py): decoder-only transformer class which autoregressively generates states of the spin chain, and computes gradients of the energy using variational Monte-Carlo 
* [train_decoder](train_decoder.py): trains the decoder by attempting to minimize the energy via gradient descent. Automatically saves model parameters and trained weights, which can then be loaded in notebooks for experiments making use of analysis and saliency modules. Executing the module requires inputting hyperparameters as cmd line arguments eg `python train_decoder.py -s 100 -d 2 -h 4 -r 2 -w 2500 -e 1 -c 10` where the parameters are 
  * s = system size
  * d = dictionary size 
  * h = number of heads
  * r = number of attention layers 
  * w = learning rate decay warmup steps 
  * e = random seed
  * c = clipping distance for relative positional encoding 
* [transformer_analysis](transformer_analysis.py): contains methods for computing correlation functions and Renyi entropies of a given transformer wavefunction. Can be imported into a separate notebook for experiments. Load trained weights and json file of model params onto an instantiated decoder and compute using provided methods.  
* [saliency_metrics](saliency_metrics.py): implementation of the saliency metrics described above. Can also be imported into a separate notebook for experiments. As above, load trained weights and model params, then compute saliency metrics with provided methods. 
