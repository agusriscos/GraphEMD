# Referencias (reducción de dimensionalidad con independencia aproximada)

## FastICA (recomendado en este script para minimizar dependencia entre salidas)

1. Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications.
   *Neural Networks*, 13(4-5), 411-430. DOI: 10.1016/S0893-6080(00)00026-5
   https://doi.org/10.1016/S0893-6080(00)00026-5

2. Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for independent component analysis.
   *IEEE Transactions on Neural Networks*, 10(3), 626-634.
   https://ieeexplore.ieee.org/document/761722

3. Documentación scikit-learn (implementación usada): `sklearn.decomposition.FastICA`
   https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

## PCA (línea base: varianza explicada, scores incorrelacionados)

4. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments.
   *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.
   https://doi.org/10.1098/rsta.2015.0202

## Nota sobre «mode mixing» en EMD

FastICA y PCA operan en el espacio de las IMF como variables instantáneas.
Reducir de *p* a *k* implica combinaciones lineales de IMF; eso no coincide con
la noción de mode mixing en EMD (separación tiempo-frecuencia). Para auditar,
revisar `metricas_reduccion.json` (correlación entre `Z`, error de reconstrucción) y `modelo_*.npz`
(matriz de mezcla / componentes, medias para `inverse_transform`).
