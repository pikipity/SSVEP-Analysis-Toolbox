.. role::  raw-html(raw)
    :format: html

The lease-square unified framework has been published in **IEEE TNSRE**: `<https://ieeexplore.ieee.org/document/10587150/>`_.

There is a Chinese introduction: `IEEE TNSRE: SSVEP识别算法的统一框架与工具箱 <http://zewang.site/blog/2024/07/IEEE%20TNSRE:%20SSVEP%E8%AF%86%E5%88%AB%E7%AE%97%E6%B3%95%E7%9A%84%E7%BB%9F%E4%B8%80%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%B7%A5%E5%85%B7%E7%AE%B1>`_.

Introduction
-------------------

In recent years, numerous brain-computer interface (BCI)
paradigms have been developed to facilitate direct communications
between human intentions and the external environment. Among these paradigms, the steady-state visual evoked
potential (SSVEP)-based BCI has become a prominent BCI
paradigm owing to its high performance, minimal training
requirements, and small individual performance differences.
The SSVEP-based BCI has been widely applied in
rehabilitation and assistive applications.

Currently, various correlation analysis (CA)-based methods
have been proposed and have achieved state-of-the-art performance
in SSVEP recognition. Nevertheless,
the relationships between these CA-based methods and the
classification/regression models have been rarely explored.

The proposed least-square (LS) framework consolidates 11 existing
CA-based SSVEP spatial filtering methods into one unified form, providing following significant contributions:

    1. The relationship between the computation of these spatial filtering and the classification/regression can be established, leading to a better understanding of the spatial filtering methods from the machine learning perspective.
    
    2. By unifying their computations, their computation strategies can be summarized and compared, facilitating a deeper exploration of relationships among existing spatial filtering approaches.
    
    3. The research gaps between these spatial filtering methods can be more easily found and filled. 

Overall, this study could provide significant
contributes to the understanding of their computation
strategies from the machine learning perspective, and holds
the potential for future developments of SSVEP recognition
methods.