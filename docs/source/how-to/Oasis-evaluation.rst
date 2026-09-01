Oasis Evaluation
================

The OasisEvaluation repository provides a streamlined way to run the Oasis stack in multi-container environment using docker-compose.
This is intended for locally testing the `OasisPlatform 1 <https://github.com/OasisLMF/OasisPlatform/tree/stable/1.23.x>`_ with a toy model example `OasisPiWind <https://github.com/OasisLMF/OasisPiWind>`_, via the Web UI `OasisUI <https://github.com/OasisLMF/OasisUI>`_.

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

.. raw:: html

    <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:1em 0">
      <iframe src="https://www.youtube.com/embed/SxRt5E-Y5Sw" title="YouTube video"
              style="position:absolute;top:0;left:0;width:100%;height:100%;border:0"
              allowfullscreen loading="lazy"></iframe>
    </div>

|

Oasis Installation Guide: Linux based OS
########################################

.. raw:: html

    <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:1em 0">
      <iframe src="https://www.youtube.com/embed/OFLTpGGEM10" title="YouTube video"
              style="position:absolute;top:0;left:0;width:100%;height:100%;border:0"
              allowfullscreen loading="lazy"></iframe>
    </div>

GitHub repository:
------------------

`Oasis Platform Evaluation <https://github.com/OasisLMF/OasisEvaluation#readme>`_.
