# MP2RAGElib (Work in progress...)
I have started this project to understand MP2RAGE more deeply by implementing Marques et al. (2010) myself. There are other implementations but I think this one is unique in the following ways:

1. It is Python.
2. It is minimalist by design (only using functions, and [Numpy](https://numpy.org/) as a core dependency).
3. The functions have extensive documentation in their docstrings.
4. It is build to teach the details of MP2RAGE T1 mapping (first to myself and maybe in the future to others :) ).
5. [TODO] It has a simple command-line interface for users with no programming or Python experience.

# Installation
### Dependencies
| Package                                            | Tested version |
|----------------------------------------------------|----------------|
| [Python](https://www.python.org/downloads/release) | 3.6            |
| [NumPy](http://www.numpy.org/)                     | 1.17.2         |


- By using pip
TODO: Register to pypi upon first release.

- By cloning this repository
```
git clone https://github.com/ofgulban/mp2ragelib.git
cd mp2ragelib
python setup.py install
```

# Usage
## Use as a library
TODO: Explore the scripts provided within `scripts` folder.

## Use from command line
TODO: Implement

## License
This project is licensed under [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause).

## References
- Marques, J. P., Kober, T., Krueger, G., van der Zwaag, W., Van de Moortele, P.-F., Gruetter, R. (2010). MP2RAGE, a self bias-field corrected sequence for improved segmentation and T1-mapping at high field. NeuroImage, 49(2), 1271–1281. <<https://doi.org/10.1016/j.neuroimage.2009.10.002>>

- [TODO] Marques, J. P., Gruetter, R. (2013). New developments and applications of the MP2RAGE sequence--focusing the contrast and high spatial resolution R1 mapping. PloS One, 8(7), e69294. <<https://doi.org/10.1371/journal.pone.0069294>>

## Acknowledgements
This project is inspired by these earlier implementations:
- [JosePMarques/MP2RAGE-related-scripts](https://github.com/JosePMarques/MP2RAGE-related-scripts)
- [Gilles86/pymp2rage](https://github.com/Gilles86/pymp2rage)
- [neuropoly/mp2rage](https://github.com/neuropoly/mp2rage)
- [spinicist/QUIT](https://github.com/spinicist/QUIT), MP2RAGE module.
