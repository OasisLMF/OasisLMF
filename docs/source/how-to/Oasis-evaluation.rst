Oasis Evaluation
================

The OasisEvaluation repository provides a streamlined way to run the Oasis stack in multi-container environment using docker-compose.
This is intended for locally testing the `OasisPlatform 1 <https://github.com/OasisLMF/OasisPlatform/tree/main-platform1>`_ with a toy model example `OasisPiWind <https://github.com/OasisLMF/OasisPiWind>`_, via the Web UI `OasisUI <https://github.com/OasisLMF/OasisUI>`_.



.. _installing_oasis:

Installing Oasis
****************

1. Install prerequisites, ``docker``, ``docker-compose``, and ``git``
2. (optional) Edit the software versions at the top of ``install.sh`` installation script, These control the oasis versions installed

|
.. code-block:: python

    export VERS_API=1.28.0
    export VERS_WORKER=1.28.0
    export VERS_UI=1.11.6
    export VERS_PIWIND='stable/1.28.x'
|

These control the oasis versions installed
 - ``VERS_API``, OasisPlatform server version
 - ``VERS_WORKER``, OasisPlatform worker version
 - ``VERS_UI``, OasisUI container version
 - ``VERS_PIWIND``, the PiWind branch to run.

3. Run the installaion script

|
.. code-block:: python

    ./install.sh
|




----

Oasis Installation Guide: Windows 10 OS
#######################################

..  youtube:: SxRt5E-Y5Sw

|
Oasis Installation Guide: Linux based OS
########################################

..  youtube:: OFLTpGGEM10



GitHub repository:
------------------

----

`Oasis Platform Evaluation <https://github.com/OasisLMF/OasisEvaluation#readme>`_.
