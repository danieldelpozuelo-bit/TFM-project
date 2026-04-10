# Towards Interpretable Twin Support Vector Machines with Probabilistic Class Membership

## Revision on TWSVM

Compared to the standard linear Support Vector Machine (SVM) model, Twin Support Vector Machine (TWSVM) solves two small quadratic programming problems (QPPs) instead of a large one. This ensures TWSVM is competitive in terms of computational time being four times faster than SVM as well as even more effective than SVM on several benchmark datasets specially when dealing with specific datasets such as the "Cross Planes" datasets.

Given a typical binary classification problem with M observations over N features, $X \in R^{MxN}$, such that an observations might belong to one of two possible classes, either class 1, $x_m = +1$ or class -1 $x_m =-1$.

Assume all the observations belonging to class 1 are included in a matrix $A \in R^{m_1xN}$ where $m_1$ is the number of observations of class 1. And $B \in R^{m_2xN}$ is a matrix including all observations on class -1.

In this context, the TWSVM algorithm seeks two non-parallel hyperplanes in the n-dimensional input space, namely, $x^Tw_1+b_1 = 0$ and $x^Tw_2+b_2 = 0$ such that, each of these hyperplanes is as close as possible to the samples of its own class and as far as possible from the samples of the other class.

Demanding the hyperplanes to be as close as possible to the samples of its own class is equivalent to minimize the sum of squared scaled distances for all the observations of a given class with respect to their corresponding hyperplane. This leads to solving the following Quadratic Programming Problems (QPPs),

$$
\begin{align*}
&\min_{w_1,b_1} \sum_{i=1}^{m_1} (x_i^Tw_1 + b_1)^2 =  \min_{w_1,b_1} \frac{1}{2}\|Aw_1+e_1b_1\|^2_2 \\
& \text{and} \\
&\min_{w_2, b_2} \sum_{j=m_1+1}^{m_2} (x_j^Tw_2 +b_2)^2 = \min_{w_2,b_2} \frac{1}{2}\|Bw_2 + e_2b_2\|_2^2
\end{align*}
$$

where, $e_i \in R^{m_i}$ represent a vector of ones. The previous QPPs define the Least- Square problems corresponding to the linear systems, $Aw_1+e_1b_1 = 0$ and $Bw_2+e_2b_2 = 0$.

Since the zero vector belongs to the column spaces of both  augmented matrices $\tilde{A} = \begin{bmatrix} A & e_1 \end{bmatrix}$, $\tilde{B} = \begin{bmatrix} B & e_2 \end{bmatrix}$ these systems always admit the trivial solution. Thus, without additional constraints, both QPPs collapse to the degenerated zero hyperplane. 

Including the following constraint on both QPPs avert it from collapsing to the trivial solution while forzing each hyperplane to be as far as possible from the observations of the opposite class,

$$
\begin{align*}
& \min_{w_1,b_1} \frac{1}{2}\|Aw_1+e_1b_1\|^2_2 \\
& \text{s.t} \\
& \begin{cases}
-(Bw_1+e_2b_1) \geq e_2
\end{cases} \\[1ex]
& \text{and} \\[1ex]
& \min_{w_2,b_2} \frac{1}{2}\|Bw_2 + e_2b_2\|_2^2 \\
& \text{s.t} \\
& \begin{cases}
Aw_2+e_1b_2 \geq e_1
\end{cases}
\end{align*}
$$

These constraints avert the hyperplanes from being collapsed into the zero hyperplane. Notice, now the trivial solution $w_i, b_i = 0$ would not be possible since it wont verify the constraints, $Aw_2 + e_1b_2 = 0 \ngeq e_1$ and $-(Bw_1+e_2b_1) = -0 \ngeq e_2$. Moreover, the constraints forces the hyperplane to be at a distance of at least 1 from the observations of the opposite class. 

For example, the constraint on the class 1 hyperplane, $-(Bw_1+e_2b_1) \geq e_2$ impose $-(x_j^Tw_1 + b_1) \geq 1\,\,\forall x_j\in B$ hence, $x_j^Tw_1+b_1 \leq -1$ requiring class 2 observations to lie on the negative side of the hyperplane, being at least 1 unit away in functional margin.

Implicitly, it has been assumed that the classes are separable in the sense that there exists two non- parallel hyperplanes satisfying the separation constraints, $-(Bw_1+e_2b_1)\geq e_2$ and $Aw_2+e_1b_2 \geq e_1$. However, perfect separability rarely holds in practice hence, the previous hard-margin formulation must be relaxed, allowing class overlapping with some observations violating the separation constraints.

Thus, to allow violations of the separation constraints non-neagative slack variables are introduced, $\{\xi_1\}_{i=1}^{m_1}$, $\{\xi_2\}_{i=m_1+1}^{m_2}$ such that, $\xi_1\geq 0,\,\, \xi_2 \geq 0$. Hence, the constraints are modified as follows,
$$
\begin{align*}
&-(Bw_1+e_2b_1) \geq e_2 - \xi_2 \\
&\text{and} \\
& (Aw_2+e_1b_2) \geq e_1 - \xi_1 \\
\end{align*}
$$

These slack variables relax the margin constraints imposed allowing some opposite class observations to be close to the hyperplane fitting the other class. 
* $\xi_i = 0$, indicates the observation verifies the one unit separation from the other class hyperplane.
* $0 < \xi_i < 1$, indicates opposite class observations falling on the correct side but close to the other class hyperplane.
* $\xi_i = 1$ indicates opposite class observations falling exactly on the wrong hyperplane 
* $\xi_2 > 1$ indicates opposite class observations crossing the wrong hyperplane.

The slack variables are incorporated into the objective function through linear penalty terms.The hyparameters $c_1, c_2$ define the trade-off between the minimization of the least-squares fitting error to the corresponding class and penalization of separation constraints violations. 

Finally, the  standard TWSVM algorithm solves the following Quadratic Programming Problem (QPP),
$$
\begin{align}
&\min_{w_1, b_1, \xi_2}\,\,\frac{1}{2}\|Aw_1+e_1b_1\|^2 + c_1e_2^T\xi_2 \tag{1} \\
&\text{s.t} \nonumber \\
&\begin{cases} 
-(Bw_1+e_2b_1)+\xi_2 \geq e_2 \\
\xi_2 \geq 0
\end{cases} \nonumber \\[2ex]
&\text{and} \nonumber \\[2ex]
&\min_{w_2, b_2, \xi_1}\,\,\frac{1}{2}\|Bw_2+e_2b_2\|^2 + c_2e_1^T\xi_1 \tag{2} \\
&\text{s.t} \nonumber \\
&\begin{cases} 
Aw_2+e_1b_2+\xi_1 \geq e_1 \\
\xi_1 \geq 0
\end{cases} \nonumber
\end{align}
$$

As stated above, opposite to classical SVM formulation, which maximizes the functional margin between classes, TWSVM minimizes the squared residuals between each observations of a given class and their associated hyperplane. Consequently, TWSVM replaces the functional margin maximization principle of standard SVM with a proximal least-squares fitting strategy, resulting in class-specific observations clustering around each hyperplane.

## TWSVM with Individual Feature Selection

This work aims to implementate a TWSVM algorithm embeded with an individual feature selection per hyperplane as in (Bai, L. Wang, Z. Shao Y.H & Deng, N. Y (2014)). The further proposed embedded feature selection method extends the $L_1$-TWSVM model that minimizes the $L_1$-norm of the feature weight vectors. As it is well known, introducing the $L_1$-norm ensures the $L_1$-TWSVM model yields sparser feature weight vectors as a result while simplifying the learning process. The $L_1$-TWSVM model is similarly to the earlier studied $L_2$-TWSVM defines two QPP as follows,

$$
\begin{align*}
&\min_{w_1, b_1, \xi_2}\;\; \|w_1\|_1 + c_1\|Aw_1+e_1b_1\|_1 + c_3e_2^T\xi_2 \\
&\text{s.t} \\
&\begin{cases}
-(Bw_1+e_2b_1) \geq e_2 - \xi_2 \\
\xi_2 \geq 0
\end{cases} \\[2ex]
&\text{and} \\[2ex]
&\min_{w_2, b_2, \xi_1}\;\;\|w_2\|_1 + c_2\|Bw_2+e_2b_2\|_1 + c_4e_1^T\xi_1 \\
&\text{s.t} \\
&\begin{cases}
(Aw_2+e_1b_2) \geq e_1 - \xi_1 \\
\xi_1 \geq 0
\end{cases}
\end{align*}
$$

Notice, this $L_1$-TWSVM could directly be used as an embedded feature selection method itself, such that whenever the two weights of a feature in the weight vectors $w_1$, $w_2$ are zeros the corresponding feature should be deleted.

The earlier representation of the $L_1$-TWSVM compose a Nondifferentiable Programming Problem due to the presence on the objective function of non linear term as, $\|w_1\|_1 = \sum_{n=1}^N|w_n^{(1)}|$ and $\|Aw_1+e_1b_1\|_1 = \sum_{i=1}^{m_1}|\sum_{n=1}^Na_{i,n}w_n^{(1)} + b_1|$ in the definition of the first TWSVM hyperplane.

This problem can be transformed into a Differentiable Linear Programming Problem (DLPP) to be solved by the simplex algorithm variant implementated within the Gurobi optimizer just introducing the following auxiliar variables, $w_i = p_i - q_i$ as well as, $Aw_1 + e_ib_1 = s_1 - t_i$ and $Bw_2 + e_2b_2 = s_2 - t_2$.

The auxiliary variables introduce a positive-negative decomposition of the weight vectors and residual terms, where each variable is expressed as the difference of two nonnegative variables. This allows their absolute values to be represented as the sum of these components, thereby enabling a linear formulation of the $L_1$-TWSVM.

$$
\begin{align*}
& \min_{p_1,q_1,s_1,t_1,b_1,\xi_2} e_1^T(p_1 + q_1) + c_1e_1^T(s_1 + t_1) + c_3e_1^T\xi_1 \\
& \text{s.t} \\
& \begin{cases}
& A(p_1 - q_1) + e_1b_1 = s_1 - t_1 \\
& -B(p_1 - q_1) + e_2b_1 \geq e_2 - \xi_2 \\
& p_1, q_1, s_1, t_1, \xi_2 \geq 0
\end{cases} \\
& \text{and} \\
& \min_{p_2,q_2,s_2,t_2,b_2,\xi_1} e_2^T(p_2 + q_2) + c_2e_2^T(s_2 + t_2) + c_4e_2^T\xi_1 \\
& \text{s.t} \\
& \begin{cases}
& B(p_2 - q_2) + e_2b_2 = s_2 - t_2 \\
& A(p_2 - q_2) + e_1b_2 \geq e_1 - \xi_1 \\
& p_2,q_2,s_2,t_2,\xi_1
\end{cases}
\end{align*}
$$

Noticing, $s_1 = A(p_1 - q_1) + e_1b_1 + t_1$ and $s_2 = B(p_2 - q_2) + e_2b_2 + t_2$ the final DLPPs to be solved by Gurobi's Simplex algorithm implementation are defined,

$$
\begin{align*}
& \min_{p_1,q_1,t_1,b_1,\xi_2} e_1^T(p_1+q_1) + c_1e_1^T(A(p_1 -q_1) + 2t_1) + c_1m_1b_1 + c_3e_1^T\xi_2 \\
& \text{s.t} \\
& \begin{cases}
& A(p_1 - q_1) + e_1b_1 + t_1 \geq 0 \\
& -B(p_1-q_1) + e_2b_1 \geq e_2 - \xi_2 \\
& p_1, q_1, t_1, \xi_2 \geq 0 
\end{cases} \\
& \text{and} \\
& \min_{p_2,q_2,t_2,b_2,\xi_1} e^T(p_2 + q_2) + c_2e^T(B(p_2 - q_2) + 2t_2) + c_2m_2b_2 + c_4e^T\xi_1 \\
& \text{s.t} \\
& \begin{cases}
& B(p_2-q_2) + e_2b_2 + t_2 \geq 0 \\
& A(p_2-q_2) + e_1b_2 \geq e_1 - \xi_1 \\
& p_2,q_2,t_2,\xi_1 \geq 0
\end{cases}
\end{align*}
$$

where $m_1 = e_1^Te_1$ and $m_2 = e_2^Te_2$ denote the total training observations of each class respectively.


## Probability model for FTWSVM (TWSVM with Feature Selection)

Once the FTSVM model has constructed both hyperplanes, a positive hyperplane $w_1^Tx+b_1$ and a negative hyperplane $w_2^Tx+b_2$, a new sample $\tilde{x}$ is assigned to either class depending on its proximity to these two hyperplanes.

Opposite to the linear decision boundary described in traditional SVM, $w^Tx +b = 0$, TWSVM defines a generally nonlinear decision boundary, $\frac{|w_1^T\tilde{x} + b_1|}{\|w_1\|} = \frac{|w_2^T\tilde{x} + b_2|}{\|w_2\|}$. In TWSVM, a testing sample $\tilde{x}$ is assigned to the positive (negative) class if its closer in perpendicular distance to the positive (negative) class hyperplane, $\frac{|w_1^T\tilde{x} + b_1|}{\|w_1\|} < \frac{|w_2^T\tilde{x} + b_2|}{\|w_2\|}$ (or $\frac{|w_1^T\tilde{x} + b_1|}{\|w_1\|} > \frac{|w_2^T\tilde{x} + b_2|}{\|w_2\|}$).

This decision rule can be equivalently expressed as a ratio such that a testing sample $\tilde{x}$ would be classified as belonging to the positive (negative) class if $\frac{|w_1^T\tilde{x}+b_1|}{|w_2^T\tilde{x} + b_2|} . \frac{\|w_2\|}{\|w_1\|} < 1$ (or $\frac{|w_1^T\tilde{x}+b_1|}{|w_2^T\tilde{x} + b_2|} . \frac{\|w_2\|}{\|w_1\|} > 1$). Thus, classification is based on comparing the relative distances of the testing sample $\tilde{x}$ to both hyperplanes and assigning it to the class of the closer one.

Shao, Y. H., Deng, N. Y., Yang, Z. M., Chen, W. J., & Wang, Z. (2012) propose a continuous scoring function to approximate the posterior probability of a sample $\tilde{x}$ belonging to a class. This score is constructed based on the geometric position of $\tilde{x}$ relative to two auxiliary hyperplanes.

Instead of working directly with the signed distances,
$$
d_1(\tilde{x}) = \frac{w_1^T\tilde{x} + b_1}{\|w_1\|},\,\,\, d_2(\tilde{x}) = \frac{w_2^T\tilde{x} + b_2}{\|w_2\|}
$$

consider a simple change of variables by using the linear combinations,
$$
d_1(\tilde{x}) - d_2(\tilde{x}),\,\,\, d_1(\tilde{x}) + d_2(\tilde{x})
$$

These new variables induce the two auxiliar hyperplanes, 
$$
\begin{cases}
d_1(\tilde{x}) - d_2(\tilde{x}) = 0 \\
d_1(\tilde{x}) + d_2(\tilde{x}) = 0
\end{cases}
$$

Notice, the first hyperplane enclose the region of samples that are equidistant from both TWSVM hyperplanes. Therefore, defining the classification boundary i.e the region of maximum uncertainty where, $P(y = +1) = \frac{1}{2}$. The second hyperplane introduces a symmetric geometric reference. It corresponds to configurations where the signed distances to the two decision hyperplanes have equal magnitude and opposite sign. It enables the normalization of distances, allowing the resulting score to be mapped into the interval [0,1]. These two auxiliary hyperplanes corresponds to the corrdinate axes in the transformed space defined by the earlier stated change of variables.

$D_+(x)$ and $D_-(x)$ are defined as the distances from sample x to each auxiliarly hyperplane,
$$
D_+(x) = \frac{|d_1(x) + d_2(x)|}{\|\frac{w_1}{\|w_1\|} + \frac{w_2}{\|w_2\|}\|} \\
D_-(x) = \frac{|d_1(x) - d_2(x)|}{\|\frac{w_1}{\|w_1\|} - \frac{w_2}{\|w_2\|}\|}
$$

Hence, one could consider two relevant quantities the minimum and maximum distances from a sample to a separating hyperplane as $d(x) = min\{D_+(x), D_-(x)\}$ and $D(x) = max\{D_+(x), D_-(x)\}$. So that we can define the ratio $\frac{d(x)}{D(x)}$.

![mapped distances](/figures/mapped_distances.png)

The axis on the earlier figure represents the distances from a sample to the positive and negative decision boundaries defined by the TWSVM model, namely $d_1(x)$, $d_2(x)$. The figure is showing how a sample's position in the original distance space spanned by ($d_1(x)$, $d_2(x)$) is mapped into the transformed coordinate system defined by ($d_1(x) - d_2(x)$, $d_1(x) + d_2(x)$). Notice, the auxiliarly hyperplanes $d_1(x) - d_2(x) = 0$ and $d_1(x) + d_2(x) = 0 $ identified in the figure by blue continuous lines corresponds to the axis of the new transformed distance space. Moreover, the distances from the sample to each of the auxiliarly hyperplanes, $D_+(x)$ and $D_-(x)$, are identified by dashed lines.


The previous figure provides the needed intuition that the greater the value of either $d(x)$ or $\frac{d(x)}{D(x)}$ for some sample, the more likely it belongs to the positive class. Hence, the proposed score function for capturing the degree of a sample belonging to the positive class was defined,

$$
f(x) = \begin{cases} d(x)(\frac{d(x)}{D(x)})^{\alpha} &\frac{|w_1^T\tilde{x} + b_1|}{\|w_1\|} < \frac{|w_2^T\tilde{x} + b_2|}{\|w_2\|}  \\
0 & \frac{|w_1^T\tilde{x} + b_1|}{\|w_1\|} = \frac{|w_2^T\tilde{x} + b_2|}{\|w_2\|} \\
-d(x)(\frac{d(x)}{D(x)})^{\alpha} & \frac{|w_1^T\tilde{x} + b_1|}{\|w_1\|} > \frac{|w_2^T\tilde{x} + b_2|}{\|w_2\|}
\end{cases}
$$

where $\alpha > 0$ is a weight parameter. Recall, the bigger the value of this function $f(x)$ the more likely the sample x belongs to the positive class. It is worthnoticing the function $f(x) \in (- \infty, \infty)$.

Finally, a probability model is constructed by using a sigmoid to transform the results yields by the earlier function into probabilities, 
$$
P(y= +1| f(x)) = \frac{1}{1 + e^{af(x) + b}} \\
p(y = -1| f(x)) = 1 - P(y = +1 | f(x))
$$

Setting the weight parameter $\alpha = 1$, the parameters $a, b$ for the sigmoid can be obtained by minimizing the cross- entropy between predicted probabilities $p_i$ and  target probabilities $t_i$ in the following convex optimization problem,

$$
\begin{align*}
\min_{a,b} \;& - \sum_{i=1}^m \Big( t_i \log(p_i) + (1-t_i) \log(1-p_i) \Big) \\
\text{s.t.} \;& p_i = P(y = +1 \mid f(x_i)), \\
&t_i = 
\begin{cases} 
\dfrac{m_1 + 1}{m_1 + 2}, & \text{if } y_i = +1,\\[1em]
\dfrac{1}{m_2 + 2}, & \text{if } y_i = -1.
\end{cases}
\end{align*}
$$

