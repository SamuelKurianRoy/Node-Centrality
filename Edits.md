> **Note:**
> Ethil Overleaf and Reply letter ond


## Overleaf

### Edit 1

Replace this part

```
Furthermore, due to numerical stability and practicality (especially in larger networks where smaller Fiedler vector components can become much smaller), it's stable to set sums of absolute differences relative to squared differences to avoid precision overflow - since keeping order (and thus, ranking and relative centrality values) is most important - but this will be discussed further later.
```

with 

```
Furthermore, due to numerical stability and practicality (especially in larger networks where smaller Fiedler vector components can become much smaller), it's stable to set sums of absolute differences relative to squared differences to avoid precision overflow. We formally characterize the relationship between this absolute-value formulation and the perturbation-derived squared formulation, including the precise conditions under which they yield identical node rankings, in Section~\ref{sec:l1_l2_justification}.
```


### Edit 2

replace this part

```
Therefore, although AEI shares mathematical components with existing 
spectral measures, its \emph{node-level formulation, perturbation-based 
derivation, dynamical interpretation, and cut-vertex sensitivity} 
collectively distinguish it as a novel centrality metric specifically 
suited to identifying structurally peripheral components for pruning.
}


To better assess the interpretative validity of the index, it can be applied to a 10-node network of a dense clique (1-4) of 4 nodes and a sparse chain of 6 nodes (5-10) - only bridged by 1 bridge node. The outcome of such an application is found in  \cref{fig:adj_edge_index_10node}.

```

with 

```
Therefore, although AEI shares mathematical components with existing 
spectral measures, its \emph{node-level formulation, perturbation-based 
derivation, dynamical interpretation, and cut-vertex sensitivity} 
collectively distinguish it as a novel centrality metric specifically 
suited to identifying structurally peripheral components for pruning.
}

{\color{blue}
\subsubsection{Theoretical Justification of the Absolute-Value Formulation}
\label{sec:l1_l2_justification}

As noted following \cref{eq:adj_edge_index}, the Adjacency Edge Index $R_i$ aggregates \emph{absolute} Fiedler-vector differences over the edges incident to node $i$, whereas the first-order perturbation expansion in \cref{eq:delta_lambda} produces a sum of \emph{squared} differences. This substitution was previously justified only informally, on grounds of numerical stability. We now make the relationship between the two formulations precise.

For node $i$ with neighbor set $N(i)$ and degree $k_i = |N(i)|$, define the perturbation-derived quantity
\begin{equation}
Q_i = \sum_{(i,j)\in E_i} (v_{2,i}-v_{2,j})^2,
\label{eq:Qi}
\end{equation}
i.e., $Q_i$ is exactly the $\varepsilon$-independent right-hand side of \cref{eq:delta_lambda}, while $R_i$ in \cref{eq:adj_edge_index} is its absolute-value analogue.

\begin{mdframed}
\textbf{Theorem 1 (Norm-equivalence bound).} For every node $i$,
\begin{equation}
Q_i \;\leq\; R_i^2 \;\leq\; k_i\, Q_i.
\label{eq:sandwich}
\end{equation}
Equivalently, $\sqrt{Q_i} \leq R_i \leq \sqrt{k_i Q_i}$.
\end{mdframed}

\textit{Proof.} Write $d_j = v_{2,i}-v_{2,j}$ for $j \in N(i)$, so $R_i = \sum_j |d_j|$ and $Q_i = \sum_j d_j^2$.

\emph{Lower bound.} Expanding the square,
\[
R_i^2 = \Big(\sum_j |d_j|\Big)^2 = \sum_j d_j^2 + \sum_{j\neq j'} |d_j||d_{j'}| \;\geq\; \sum_j d_j^2 = Q_i,
\]
since the cross terms are non-negative.

\emph{Upper bound.} By the Cauchy--Schwarz inequality applied to the vectors $(|d_j|)_{j\in N(i)}$ and $(1)_{j\in N(i)}$ in $\mathbb{R}^{k_i}$,
\[
R_i^2 = \Big(\sum_j |d_j|\cdot 1\Big)^2 \;\leq\; \Big(\sum_j d_j^2\Big)\Big(\sum_j 1^2\Big) = k_i\, Q_i. \qquad \blacksquare
\]

Equality on the left holds iff at most one neighbor of $i$ has $d_j \ne 0$; equality on the right holds iff $|d_j|$ is constant across all neighbors of $i$. The ratio $R_i^2/Q_i \in [1, k_i]$ is therefore governed entirely by how concentrated or uniform node $i$'s neighbor-wise Fiedler gaps are, not by an unconstrained modeling choice: the absolute-value formulation is a bounded, fully characterized deformation of the perturbation-exact quantity.

\begin{mdframed}
\textbf{Corollary 1 (Sufficient condition for rank agreement).} For two nodes $i,j$, define the interval $I_i = \big[\sqrt{Q_i},\, \sqrt{k_i Q_i}\,\big]$. If $I_i$ and $I_j$ are disjoint, then $R_i$ and $Q_i$ induce the same relative order between $i$ and $j$, irrespective of the internal shape of either node's Fiedler-gap vector.
\end{mdframed}

\textit{Proof.} If $I_i$ lies entirely below $I_j$ (i.e.\ $\sqrt{k_i Q_i} \le \sqrt{Q_j}$), then by \cref{eq:sandwich}, $R_i \le \sqrt{k_i Q_i} \le \sqrt{Q_j} \le R_j$, so $R_i \le R_j$ and $Q_i \le Q_j$ agree (and symmetrically for the reverse case). $\blacksquare$

\textbf{Empirical validation.} We tested both \cref{eq:sandwich} and the practical extent of rank disagreement on the 198-node Jazz Musicians network introduced in Section 4.1 -- an unweighted graph for which $R_i$ is computed exactly as in \cref{eq:adj_edge_index}, with no edge-weighting confound. The bound in \cref{eq:sandwich} holds exactly for all 198 nodes, as Theorem 1 guarantees. The two rankings are strongly, but not perfectly, correlated: Spearman's $\rho = 0.927$ ($p < 10^{-80}$) and Kendall's $\tau = 0.775$ between $R_i$ and $Q_i$ across all nodes. Of the $\binom{198}{2}=19{,}503$ pairwise node comparisons, $32.6\%$ are \emph{guaranteed} to agree by Corollary 1 alone, without computing $R_i$ at all; among the remaining ambiguous pairs, the two criteria still agree empirically $83.3\%$ of the time.

At the specific operating points used for pruning (removing the lowest-scoring 20\%, 30\%, and 40\% of nodes, as in Section 4.1 and Section 4.2), the overlap between the $R$-selected and $Q$-selected pruning sets is $72.5\%$, $84.7\%$, and $84.8\%$, respectively. The disagreements are not unstructured noise: they follow exactly the mechanism predicted by Theorem 1. Nodes that $R_i$ flags as peripheral but $Q_i$ does not have a median degree of $6$ (versus a network-wide median of $25$), while nodes that $Q_i$ flags as peripheral but $R_i$ does not have a median degree of $34$. This is the expected behaviour of an unnormalized $\ell_1$ aggregate: because $R_i$ sums, rather than averages, over $k_i$ neighbours, its value grows mechanically with degree relative to $Q_i$, biasing low-degree nodes toward appearing more peripheral and high-degree nodes toward appearing more central than the perturbation-exact criterion would rank them.

We therefore make a more precise and falsifiable claim than our original submission: the absolute-value formulation does \emph{not} preserve the perturbation-derived ranking exactly. What can be substantiated is a provable two-sided bound (Theorem 1) together with a quantitative, mechanistically explained characterization of where and why the two criteria diverge (Corollary 1 and the degree analysis above). Rank disagreement between $R_i$ and $Q_i$ is attributable specifically to degree heterogeneity, not to an unconstrained modeling choice, and is largely confined to comparisons between nodes of very different degree. We retain the absolute-value formulation $R_i$ for the numerical-stability reasons stated above and for its analytical tractability in the cut-vertex setting just discussed, while reporting this bound so that its relationship to the underlying perturbation theory is explicit and falsifiable rather than asserted.
}


To better assess the interpretative validity of the index, it can be applied to a 10-node network of a dense clique (1-4) of 4 nodes and a sparse chain of 6 nodes (5-10) - only bridged by 1 bridge node. The outcome of such an application is found in  \cref{fig:adj_edge_index_10node}.

```


### Edit 3

Replace this part

```
\textbf{Formal property.} We state the following property that 
concretely distinguishes AEI from global spectral measures:

\begin{mdframed}
\textbf{Proposition (Cut-vertex sensitivity).} Let $G$ be a connected 
graph and let $v$ be a cut-vertex of $G$ (i.e., its removal disconnects 
$G$). Then $R_v > R_u$ for any node $u$ whose removal leaves $G$ 
connected, provided the Fiedler vector components on either side of the 
cut are non-degenerate.
\end{mdframed}

\textit{Justification.} A cut-vertex $v$ bridges two components with
```

with


```
\textbf{Empirical observation.} We note the following empirical pattern, which concretely 
distinguishes AEI from global spectral measures in the cases we have examined (a general proof under precise structural assumptions, e.g.\ on spectral gap and cut-edge count, is left to future work):

\begin{mdframed}
Let $G$ be a connected 
graph and let $v$ be a cut-vertex of $G$ (i.e., its removal disconnects 
$G$). In the networks studied here, $R_v > R_u$ for nodes $u$ whose removal leaves $G$ 
connected, whenever the Fiedler vector components on either side of the 
cut are non-degenerate.
\end{mdframed}

\textit{Empirical support.} A cut-vertex $v$ bridges two components with
```

> **Note:**
> Evidam Muthal Ollathu Reply Letter aa

## ***Reply to Reviewer 2, point 1***

Replace this in R4.tex (Nammadu Reply letter atu alle?)

```
\item One scientific weakness appears to be that AEI is not truly derived from the perturbative formula, which would lead to a direct dependence on $\sum(v_i-v_j)^2$ or similar quantities. The transformation from square to absolute value is justified with arguments of numerical stability and ranking. However, it is not formally proved, nor is the ranking equivalence demonstrated. The best solution here would be to provide a theorem.
\begin{indentedenv}
    \begin{itemize}
        \textcolor{blue}{We agree with the Reviewer 
}
    \end{itemize}
\end{indentedenv}
```

with 


```
\item One scientific weakness appears to be that AEI is not truly derived from the perturbative formula, which would lead to a direct dependence on $\sum(v_i-v_j)^2$ or similar quantities. The transformation from square to absolute value is justified with arguments of numerical stability and ranking. However, it is not formally proved, nor is the ranking equivalence demonstrated. The best solution here would be to provide a theorem.
\begin{indentedenv}
    \begin{itemize}
        \textcolor{blue}{We agree with the Reviewer that this transition was not rigorously justified in the original submission. We have added a new subsection (Section~2.2.1, ``Theoretical Justification of the Absolute-Value Formulation'') that proves a formal norm-equivalence theorem bounding the absolute-value AEI, $R_i$, against the perturbation-derived squared quantity $Q_i = \sum_{j\in N(i)}(v_{2,i}-v_{2,j})^2$: specifically, $Q_i \le R_i^2 \le k_i Q_i$, where $k_i$ is the degree of node $i$ (Theorem~1). We further derive a corollary giving an explicit, checkable sufficient condition under which the two rankings are provably identical for a given pair of nodes (Corollary~1). We validate this empirically on the Jazz Musicians network used elsewhere in the manuscript: the two rankings have Spearman's $\rho = 0.927$ and Kendall's $\tau = 0.775$, and the bottom-20\%/30\%/40\% pruning sets selected by $R_i$ overlap with those selected by $Q_i$ at 72.5\%/84.7\%/84.8\%, respectively. We further show that the residual disagreement is not unstructured noise but is attributable specifically to degree heterogeneity, exactly as predicted by Theorem~1: nodes where the two criteria disagree have systematically atypical degree relative to the network median. Rather than asserting the two formulations are interchangeable, we now state precisely how they relate, with a formal two-sided bound and a quantified, mechanistically explained account of where and why they can diverge.}
    \end{itemize}
\end{indentedenv}

```


### Reply to Reviewer 3, point 1

Replace this 

```
\item First, the theoretical foundation of AEI needs to be clarified and strengthened. The manuscript derives a first-order perturbation expression involving the sum of squared Fiedler-vector differences, but then defines AEI using the sum of absolute differences. The claim that this change mainly preserves ranking is not generally valid: squared and absolute aggregations can produce different node rankings. The authors should either use the perturbation-derived squared formulation, provide a rigorous justification for the absolute-value formulation, or empirically demonstrate that the ranking is stable under this change. In addition, the stated cut-vertex sensitivity property appears too strong and is currently justified only intuitively. It should either be formally proven under precise assumptions or rewritten as an empirical observation rather than a general proposition.
\begin{indentedenv}
    \begin{itemize}
        \textcolor{blue}{We agree with the Reviewer 
}
    \end{itemize}
\end{indentedenv}
```

with 


```
\item First, the theoretical foundation of AEI needs to be clarified and strengthened. The manuscript derives a first-order perturbation expression involving the sum of squared Fiedler-vector differences, but then defines AEI using the sum of absolute differences. The claim that this change mainly preserves ranking is not generally valid: squared and absolute aggregations can produce different node rankings. The authors should either use the perturbation-derived squared formulation, provide a rigorous justification for the absolute-value formulation, or empirically demonstrate that the ranking is stable under this change. In addition, the stated cut-vertex sensitivity property appears too strong and is currently justified only intuitively. It should either be formally proven under precise assumptions or rewritten as an empirical observation rather than a general proposition.
\begin{indentedenv}
    \begin{itemize}
        \textcolor{blue}{We agree with the Reviewer on both points.

(i) Squared vs. absolute-value aggregation: as for Reviewer~2's related comment, we have added Section~2.2.1, which proves a norm-equivalence theorem, $Q_i \le R_i^2 \le k_i Q_i$ (Theorem~1), bounding the absolute-value AEI $R_i$ against the perturbation-derived quantity $Q_i = \sum_{j\in N(i)}(v_{2,i}-v_{2,j})^2$, together with a corollary giving a checkable sufficient condition for exact rank agreement between the two criteria (Corollary~1). We validate this on the Jazz Musicians network (Spearman's $\rho = 0.927$; 72.5--84.8\% pruning-set overlap across the three sparsity levels used in this study) and show that residual disagreement is concentrated among nodes with atypical degree, exactly as the theorem predicts. We no longer claim the two formulations are interchangeable; we instead state precisely how they relate and quantify where and why they can diverge.

(ii) Cut-vertex sensitivity: we agree that our original Proposition overstated what we could support with the evidence presented. We have relabeled it as an empirical observation rather than a general proposition, and revised the surrounding text accordingly, since we do not currently have a formal proof under precise structural assumptions (e.g., conditions on the spectral gap and the number of cut edges) and prefer to report only what our experiments directly support.}
    \end{itemize}
\end{indentedenv}

````