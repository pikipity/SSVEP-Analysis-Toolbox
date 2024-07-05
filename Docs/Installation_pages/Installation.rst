.. _installation-page:

Installation
==============================

Install from pip
------------------

If you have not installed this toolbox, you can install it by

``pip install SSVEPAnalysisToolbox``

If you already installed this toolbox, you can update it by 

``pip install --upgrade --force-reinstall SSVEPAnalysisToolbox``

Note: This toolbox does not contain `jupyter`. You may check `jupyter official website <https://jupyter.org/install>`_ to install `jupyter` and run some demos.


Directly download from Github
-------------------------------------

1. Donwload or clone `this repository <https://github.com/pikipity/SSVEP-Analysis-Toolbox.git>`_.
   
   ``git clone https://github.com/pikipity/SSVEP-Analysis-Toolbox.git``

2. Install required packages. You can find all required packages in :file:`environment.yml`. If you use `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ or `Anaconda <https://www.anaconda.com/>`_, you can create a new virtual environment using :file:`environment.yml`.
   
   ``conda env create -f environment.yml``

3. Now, you can use this toolbox. If you use ``conda`` or ``anaconda``, do not forget to enter your virtual environment. In addition, when you use this toolbox, remember add the toolbox's path in your python searching path. You can add the follow code in your python file.

    .. code:: ipython3

        import sys
        sys.path.append(<toolbox_path>)
