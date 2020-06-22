# denoising-autoencoder
3層のデノイジングオートエンコーダをnumpyのみを用いて実装しました．  
偏微分の計算はライブラリなどを使わずに，微分の連鎖律を手で計算することで導出しました．  
今回は学習を，層数が3層，ノード数が[100,80,100]，データ数が50，データ長が100，学習係数を10，学習回数を10000，という条件の下で行いました． 
結果はsrc/resultの中にあります． 

I implemented a three-layer Denoising Auto-encoder using only numpy.  
The partial derivative was derived by computing the chain law of the derivative by hand, without using a library.  
This time, the learning was performed under the conditions that the number of layers is 3, the number of nodes is [100,80,100], the number of data is 50, the data length is 100, the learning coefficient is 10, and the number of learning is 10000.  
You can find the result in src/result.
