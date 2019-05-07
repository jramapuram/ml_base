# ml_base : A base to startup your VAE or classifier projects

ml_base is intended to be used as a starting point for quick prototyping of Variational Autoencoder or classifier / regressor projects


## Usage Classifier/Regressor Project

First go create your repository to house your project (`github.com/YOUR_USERNAME/YOUR_PROJECT.git` below).

``` bash
git clone recursive git+ssh://git@github.com/jramapuram/ml_base.git                # clone the repo
git remote set-url origin git+ssh://git@github.com/YOUR_USERNAME/YOUR_PROJECT.git  # change the endpoint

# Prototype the idea you want

git push -f                                                                        # push to new remote
```


## Usage VAE Project

First go create your repository to house your project (`github.com/YOUR_USERNAME/YOUR_PROJECT.git` below).

``` bash
git clone recursive git+ssh://git@github.com/jramapuram/ml_base.git                # clone the repo
git clone recursive git+ssh://git@github.com/jramapuram/vae.git models             # clone the VAE repo
git remote set-url origin git+ssh://git@github.com/YOUR_USERNAME/YOUR_PROJECT.git  # change the endpoint

# Prototype the idea you want

git push -f                                                                        # push to new remote
```


### Example Convolutional VAE Usage

``` bash
python vae_main.py --vae-type=simple --debug-step --disable-gated --reparam-type=isotropic_gaussian
```


### Example VRNN Usage

``` bash
python vae_main.py --vae-type=vrnn --debug-step --disable-gated --reparam-type=isotropic_gaussian
```
