.. role::  raw-html(raw)
    :format: html

Example 1: How to use basic APIs
------------------------------------

This example shows the basic functions of this toolbox, including:

+ How to initialize a dataset, and how to hook required functions.
+ How to initialize a recognition model.
+ How to get the training data from the dataset, and train the model.
+ How to get the testing data from the dataset, and test the model.
+ How to calculate the classification accuracy and the ITR.

You can find the related code in :file:`demo/simple_example.py` or :file:`demo/simple_example.ipynb`.

First, We need to add the toolbox into the search path.

.. code:: ipython3

    import sys
    sys.path.append('..')

Then, before we start the simulation, we need to prepare the dataset. If
there is not data in the given path, the dataset API will automatically
download the correspnding data. In this example, we will use the
Benchmark Dataset as example.

.. code:: ipython3

    from SSVEPAnalysisToolbox.datasets import BenchmarkDataset
    dataset = BenchmarkDataset(path = '2016_Tsinghua_SSVEP_database')


.. parsed-literal::

    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S1.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S1.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: 2e72ca82202ad82268c45a204cc19ae9df6e4866c62b4ecd1e2a367ec14f601e
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S2.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S2.mat.7z'.
     53%|####################1                 | 56.0M/105M [00:15<00:13, 3.55MB/s]
    C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\..\SSVEPAnalysisToolbox\datasets\basedataset.py:174: UserWarning: There is an error when donwloading 'S2' data. So retry (1) after 10 seconds.
      warnings.warn("There is an error when donwloading '{:s}' data. So retry ({:n}) after 10 seconds.".format(subject.ID, download_try_count))
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S2.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S2.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 105GB/s]
    SHA256 hash of downloaded file: 119811b9d12f6fb865b4e9436f688208344405282d9621a1691349972f0f73fa
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S3.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S3.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 107GB/s]
    SHA256 hash of downloaded file: 3c6addba14c5515f60d2efa8d2c7449f3f6a25fba735bb7d3861cd3187504e83
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S4.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S4.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 104GB/s]
    SHA256 hash of downloaded file: 6ba7ae965ac5f1c3df08c5e1497c3f48f03deb112078521d580ee484d3d714a6
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S5.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S5.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: 120e6b5f7822095484730f0c180cea2c71503f4fa0bf47003d14db5c1b524566
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S6.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S6.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: 8ed4971187a07b6223cc1f0739bd5bbe6501b15ad231cc0597debba4a423cd3f
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S7.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S7.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 107GB/s]
    SHA256 hash of downloaded file: dea154a7b4a025ae645bf812cda77db3709e713813aefabc8babe029c87aa28e
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S8.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S8.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 105GB/s]
    SHA256 hash of downloaded file: e8eec0350ee02d6d5f8db44c316be16ad839db728446de077f263fad1ee57ab4
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S9.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S9.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 105GB/s]
    SHA256 hash of downloaded file: 45ec6d9798b5dab85c3a4953f393d6d762e95eac44bf1b981947b7449e0ae4f8
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S10.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S10.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: a21e8caa427d99405bcca8da58f174dd26872958afcb2b3f7c04e6c2ec2f2b18
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S11.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S11.mat.7z'.
     93%|###################################4  | 99.8M/107M [00:16<00:01, 6.22MB/s]
    C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\..\SSVEPAnalysisToolbox\datasets\basedataset.py:174: UserWarning: There is an error when donwloading 'S11' data. So retry (1) after 10 seconds.
      warnings.warn("There is an error when donwloading '{:s}' data. So retry ({:n}) after 10 seconds.".format(subject.ID, download_try_count))
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S11.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S11.mat.7z'.
    100%|########################################| 107M/107M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: a2c70968f10292ca66cea826e773c3e3572c1aae7e2c00227f8e322245ce6929
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S12.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S12.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 105GB/s]
    SHA256 hash of downloaded file: 59d6e8aa6623039ad486a69e56cca9ef6d9a58872c80ee8652935011b38bea0f
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S13.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S13.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: 0b927e61ccfdaea86bee58ed65fb09009ce78ce81ba71a75f6bbac31dc7e33f7
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S14.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S14.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: 466f52b7eb37042d8970690f3801aceccead00539cfde1c2e1d9e6d239e5ef43
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S15.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S15.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 103GB/s]
    SHA256 hash of downloaded file: a38ed9908c7d139cfc2ff7a07858a6e442eada02dfa2188e222ef2e4dcaa9b75
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S16.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S16.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 105GB/s]
    SHA256 hash of downloaded file: 2bf93eb3229ebe25d4065e892f5f3be1d447e612c5707b3c5d7f2b3aba8e1d29
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S17.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S17.mat.7z'.
     50%|###################                   | 53.1M/106M [00:15<00:15, 3.38MB/s]
    C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\..\SSVEPAnalysisToolbox\datasets\basedataset.py:174: UserWarning: There is an error when donwloading 'S17' data. So retry (1) after 10 seconds.
      warnings.warn("There is an error when donwloading '{:s}' data. So retry ({:n}) after 10 seconds.".format(subject.ID, download_try_count))
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S17.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S17.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: d2776bb79ed0f13bf1d7fb576351089d34e6245bad9f85cb220932b49d4aa02f
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S18.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S18.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: 5a09b481424897f0dfef13cd373eacc89ef62c3727580b35d5a2e45dcdc57d9e
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S19.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S19.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 105GB/s]
    SHA256 hash of downloaded file: 443b867d933c03ed7a799f3705ce09ea393c5d7cb6be79c1858e361b44b37f8d
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S20.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S20.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 108GB/s]
    SHA256 hash of downloaded file: 4393d336d841b20d0e06875ec0dace2c9e2918e3ef066ab1860657bb11cad2e9
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S21.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S21.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 105GB/s]
    SHA256 hash of downloaded file: 3e2ab753e8708398e940548e656253cae6531b2c6a2842c33fc3e5dcf8d373db
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S22.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S22.mat.7z'.
     14%|#####2                                 | 14.2M/105M [00:14<01:35, 949kB/s]
    C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\..\SSVEPAnalysisToolbox\datasets\basedataset.py:174: UserWarning: There is an error when donwloading 'S22' data. So retry (1) after 10 seconds.
      warnings.warn("There is an error when donwloading '{:s}' data. So retry ({:n}) after 10 seconds.".format(subject.ID, download_try_count))
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S22.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S22.mat.7z'.
    100%|###############################################| 105M/105M [00:00<?, ?B/s]
    SHA256 hash of downloaded file: fda0f0963334f09175ea14b1683b1af542624bf37d0e1010d4ca9f2400a671b9
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S23.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S23.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 105GB/s]
    SHA256 hash of downloaded file: 0b08a6c2562782654b072b935e6dc7248bfe221f2139e94b6afdf8c4c8a87c3f
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S24.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S24.mat.7z'.
     60%|######################9               | 63.8M/106M [00:16<00:11, 3.76MB/s]
    C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\..\SSVEPAnalysisToolbox\datasets\basedataset.py:174: UserWarning: There is an error when donwloading 'S24' data. So retry (1) after 10 seconds.
      warnings.warn("There is an error when donwloading '{:s}' data. So retry ({:n}) after 10 seconds.".format(subject.ID, download_try_count))
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S24.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S24.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: c137d2c20aa8ed94b00f0adcf1cbd49441155ad37d871280028f1437415524a4
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S25.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S25.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: 16e99d0eeb23a45b609b944a57faba9ee411fcf63c9ee3ebbc4451e8bcf47693
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S26.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S26.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 107GB/s]
    SHA256 hash of downloaded file: 49385de18c85724d9b1aecfc51b18d2b2c250a9dcd69f95261b51fea7afe4392
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S27.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S27.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 107GB/s]
    SHA256 hash of downloaded file: 0d4ad7b103b00d534f5fa4d0305df5f713feae08547ddf9da055f3a6499fc6fd
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S28.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S28.mat.7z'.
    100%|#######################################| 106M/106M [00:00<00:00, 52.9GB/s]
    SHA256 hash of downloaded file: 5edbbb09f520eef44f3ac4afdcfcdbebc0f8eb2df68ac58bde87b1cbf9732453
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S29.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S29.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 105GB/s]
    SHA256 hash of downloaded file: 32e2beac9a21a26e2f63c37d9a8dcfd2e052c9649009a44108e2ade4047888be
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S30.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S30.mat.7z'.
    100%|###############################################| 106M/106M [00:00<?, ?B/s]
    SHA256 hash of downloaded file: d9848153d502412ae0a3204e4b3b8087bd9a03dac24c5d8c68e7374bdad1e556
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S31.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S31.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 107GB/s]
    SHA256 hash of downloaded file: c68e7e18422fa24700ffd1b02d0c062c10d4e2cf6711509507aad37d3dd9b239
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S32.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S32.mat.7z'.
    100%|########################################| 105M/105M [00:00<00:00, 105GB/s]
    SHA256 hash of downloaded file: 93a17de4989ad4e42823236a24338ca8b3a44e1e688a7c0f9b378208dc69ebd0
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S33.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S33.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: cc1fd968bc8cda13374a93bfdf6caa04b3e178af6d822cb7873d46868cc23a12
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S34.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S34.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 106GB/s]
    SHA256 hash of downloaded file: 6c92839da0cf5b8084d06cd33191bbf50a3f323c17fc9c1ce2f4a2644e02b03f
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/S35.mat.7z' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\S35.mat.7z'.
    100%|########################################| 106M/106M [00:00<00:00, 108GB/s]
    SHA256 hash of downloaded file: b4d821150a1f2eb808dec077ae6769ba0fd6bdb0077c94ea07c575240324de77
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/Readme.txt' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\Readme.txt'.
    0.00B [00:00, ?B/s]
    SHA256 hash of downloaded file: 3bf106f1901a2ce2c7c309fee948eb13a692597a00e23cbc8d8b9ae170988e69
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/Sub_info.txt' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\Sub_info.txt'.
    0.00B [00:00, ?B/s]
    SHA256 hash of downloaded file: 5b5e833c438a169aca86cbabb99d0509ff3b1ca1d9c7de04ab54874a089a2d17
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/64-channels.loc' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\64-channels.loc'.
    100%|#####################################| 1.98k/1.98k [00:00<00:00, 2.00MB/s]
    SHA256 hash of downloaded file: da8c1d84451930234392b9283fccffb7994d69ed97bb452c6927613bb33c3ab0
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    Downloading data from 'http://bci.med.tsinghua.edu.cn/upload/yijun/Freq_Phase.mat' to file 'C:\Users\wangz\Documents\GitHub\SSVEP-Analysis-Toolbox\demo\2016_Tsinghua_SSVEP_database\Freq_Phase.mat'.
    100%|##########################################| 366/366 [00:00<00:00, 363kB/s]
    SHA256 hash of downloaded file: 6059f712688ec9e5df0beace9244dc0a4b03c418dacdef86bac50cf2b95f71b5
    Use this value as the 'known_hash' argument of 'pooch.retrieve' to ensure that the file hasn't changed if it is downloaded again in the future.
    

Because EEG signals normally contain large noise, we need do
preprocesses when we extract signals. Therefore, we need hook the
preprocess method on the dataset. The Benchmark Dataset paper already
provides the suggested preprocess methods. These method has been
included in this toolbox and can be directly used.

.. code:: ipython3

    from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import preprocess
    dataset.regist_preprocess(preprocess)

Because the filter-bank approach has been successfully adopted to
improve the recognition performance in literature, we need to hook the
filter-bank method on the dataset. The Benchmark Dataset paper already
provides the suggested filter-bank method. This method has also been
included in this toolbox and can be directly used.

.. code:: ipython3

    from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import filterbank
    dataset.regist_filterbank(filterbank)

After preparing the dataset, we need to prepare the recognition method.
The toolbox contains various methods with different implementations.
This example use the eCCA method as an example to show how to use the
method API. In addition, because we use the filter-bank approach, we
need to predefine the weights of different filter banks. The Benchmark
Dataset paper already provides the suggested weights. The method of
generating these weights has been implemented in this toolbox and can be
directly used.

.. code:: ipython3

    from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import suggested_weights_filterbank
    weights_filterbank = suggested_weights_filterbank()
    from SSVEPAnalysisToolbox.algorithms import ECCA
    recog_model = ETRCA(weights_filterbank = weights_filterbank)

Now, we can prepare the simulation. In this example,

1. we will only use 9 occipital channels;
2. All 40 classes in the Benchmark data are considered.
3. 5 harmonic components are considered in the SSVEP reference signals;
4. The first 1 second EEG signals after removing 0.14s latency are
   applied for this example;
5. Only the second subject’s EEG is used for the individual recognition;
6. EEG signals in the first block is used for testing the recognition
   method;
7. EEG signals in other blocks is used for training the recognition
   method.

.. code:: ipython3

    from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import suggested_ch
    ch_used = suggested_ch()
    all_trials = [i for i in range(dataset.trial_num)]
    harmonic_num = 5
    tw = 1
    sub_idx = 2-1
    test_block_idx = 0
    test_block_list, train_block_list = dataset.leave_one_block_out(block_idx = test_block_idx)

The whole simulation is divided into 2 steps:

1. Train the recognition model:

   1. Prepare the training materials: The training process of most
      recognition methods requires the training data, corresponding
      labels, the SSVEP reference signals (sine-cosine reference
      signals), and freqeucies of labels. Although the eCCA does not
      need freqeucies of labels, we still show how to prepare and input
      them.
   2. Use the training materials to train the model. We also show how to
      record the training time.

.. code:: ipython3

    ref_sig = dataset.get_ref_sig(tw, harmonic_num)
    freqs = dataset.stim_info['freqs']
    X_train, Y_train = dataset.get_data(sub_idx = sub_idx,
                                        blocks = train_block_list,
                                        trials = all_trials,
                                        channels = ch_used,
                                        sig_len = tw)

.. code:: ipython3

    import time
    tic = time.time()
    recog_model.fit(X=X_train, Y=Y_train, ref_sig=ref_sig, freqs=freqs) 
    toc_train = time.time()-tic

2. Test the recognition model:

   1. Prepare the testing materials: Normally, we only need the testing
      EEG signals. But we also extract the corresponding testing labels
      for further calculating classification accuracy;
   2. Use the testing materials to test the model. We also record the
      testing time and compute the averaged testing time of each trial
      for further calculating the ITR.

.. code:: ipython3

    X_test, Y_test = dataset.get_data(sub_idx = sub_idx,
                                        blocks = test_block_list,
                                        trials = all_trials,
                                        channels = ch_used,
                                        sig_len = tw)

.. code:: ipython3

    tic = time.time()
    pred_label, _ = recog_model.predict(X_test)
    toc_test = time.time()-tic
    toc_test_onetrial = toc_test/len(Y_test)

Finally, we can use the build-in functions to quickly calculate the
classification accuracy and ITR.

.. code:: ipython3

    from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr
    acc = cal_acc(Y_true = Y_test, Y_pred = pred_label)
    itr = cal_itr(tw = tw, t_break = dataset.t_break, t_latency = dataset.default_t_latency, t_comp = toc_test_onetrial,
                  N = len(freqs), acc = acc)
    print("""
    Simulation Information:
        Method Name: {:s}
        Dataset: {:s}
        Signal length: {:.3f} s
        Channel: {:s}
        Subject index: {:n}
        Testing block: {:s}
        Training block: {:s}
        Training time: {:.5f} s
        Total Testing time: {:.5f} s
        Testing time of single trial: {:.5f} s
    
    Performance:
        Acc: {:.3f} %
        ITR: {:.3f} bits/min
    """.format(recog_model.ID,
               dataset.ID,
               tw,
               str(ch_used),
               sub_idx,
               str(test_block_list),
               str(train_block_list),
               toc_train,
               toc_test,
               toc_test_onetrial,
               acc*100,
               itr))


.. parsed-literal::

    
    Simulation Information:
        Method Name: eTRCA
        Dataset: Benchmark Dataset
        Signal length: 1.000 s
        Channel: [47, 53, 54, 55, 56, 57, 60, 61, 62]
        Subject index: 1
        Testing block: [0]
        Training block: [1, 2, 3, 4, 5]
        Training time: 0.08602 s
        Total Testing time: 1.27929 s
        Testing time of single trial: 0.03198 s
    
    Performance:
        Acc: 97.500 %
        ITR: 180.186 bits/min
    
    


