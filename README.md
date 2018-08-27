MovingLineAGIP
==============

.. contents:: **Table of Contents**
    :backlinks: none

Installation
------------

MovingLineAGIP is distributed on `PyPI <https://pypi.org>`_ as a universal
wheel and is available on Linux/macOS and Windows and supports
Python 2.7/3.5+ and PyPy.

    $ pip install MovingLineAGIP
    
Warnings
-------

 * **Greatly reduced motion detection.** The standard vision filter and "RGC" tensor only mimics midget cells, not 
 parasol cells. To mimic parasol cells for better motion, two tensors are needed: one for the center-surround cetection
 of whatever size you desire, and another to supress cells more as they continue firing. These need not detect full 
 color to match human vision though, as retinal motion detection is largely color blind in humans.
 
 * **Will not detect shinyness or other stereoscopic effects.** Even with two cameras. The human brain overlaps visual 
 input in specific ways that need to be mimicked even for seemingly simple tasks like recognizing an object as shiny or 
 glittery. The human brain does this by putting the left side of vision of both eyes in the left cortex, and the right 
 side of vision of both eyes in the right cortex. This means both cortexes should be activated even when one eye is 
 closed. Instead, these effects can be detected with motion.
 
 * **Will only detect striped edges.** Currently, the boundary detector tensor only generates a tensor similar to a 
 tripartite stripe detector simple cell. This means there is a small chance edges with no border may not be detected.

Troubleshooting
---------------

###Cannot install with pip
Only Python version 2.7, 3.3, 3.4, 3.5, and 3.6 are currently supported. You need to make sure you're using one of those.

License
-------

MovingLineAGIP is distributed under the terms of both

- `MIT License <https://choosealicense.com/licenses/mit>`_
- `Apache License, Version 2.0 <https://choosealicense.com/licenses/apache-2.0>`_

at your option.
