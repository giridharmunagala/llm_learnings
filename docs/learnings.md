### Jacard Similarity
Measures how similar two sets are - Defined as Fraction of shared elements in two sets. Also

> Probability of a randomly selected element from the union of the sets to also being in the intersection.

[Giorgi blog - Minhash explanation](https://giorgi.tech/blog/minhashing/)

**Ex:**
```
Set A: {1,2,3}
Set B: {3,4,5}
Jacard similarity = no. of shared elements / no. of total elements. = 1 / 5 = 0.2
```

### Min Hash - Faster way of computing the Jacard Similarity - Approximation

Since computing this Jacard similarity requires a lot of compute - we need nC2 comparisons (n - no of sets) & for each set we require len(A) + len(B) operations for each combination.

Approximation is re-writing this Jacord similarity into a matrix form & some operations as listed below. 

Lets say there are 3 sets
```
Set A: {1,2,3}
Set B: {3,4,5}
Set C: {1,5,6}
```

**Step 1:** Create a adjacency like matrix which indicates the associations

| Elements / Sets| S_A | S_B | S_C |
|----------|-----|-----|-----|
| 1 | 1 | 0 | 1 |
| 2 | 1 | 0 | 0 |
| 3 | 1 | 1 | 0 |
| 4 | 0 | 1 | 0 |
| 5 | 0 | 1 | 1 |
| 6 | 0 | 0 | 1 |

**Step 2:** Shuffle it by shuffling the rows
| Elements / Sets| S_A | S_B | S_C |
|----------|-----|-----|-----|
| 6 | 0 | 0 | 1 |
| 1 | 1 | 0 | 1 |
| 3 | 1 | 1 | 0 |
| 5 | 0 | 1 | 1 |
| 4 | 0 | 1 | 0 |
| 2 | 1 | 0 | 0 |

**Step 3:** For each column compute the firstelement corresponding to the non-zero value through a mappper m.
```
m(S_A) = 1 
m(S_B) = 3 
m(S_C) = 6
```
**Step 4:** Compute the approximate Jacards similarity b/w two sets as 
$\hat{JS}(S_i,S_j)=\begin{cases}1 & \text{if } m(S_i)=m(S_j) \\ 0 & \text{otherwise}\end{cases}$

$Pr[m(S_i)=m(S_j)]= E[\hat{JS}(S_i,S_j)] = JS(S_i,S_j)$

So to get the real $JS(S_i,S_j)$ we need to iterate over many such suffling operations & the approximation gets better as iterations increase. 

This is still an expensive operation as shuffling long sequences is hard. So Min hash does this in 1 go by having inherent shuffling through hashing functions & only performs 1 pass over the data.

**Step 1:**
Pick K hash functions from a hash family at random which maps the elements from the sets into hash values.

**Step 2:**
```
j = hash function index; K = total no. of hash functions;
S: Current set of values;
Initialize  V_j = infinity

for element in S:
    for j in K:
        if h_j(element) < V_j:
            v_j = h_j(element)

Then m_j(S) = v_j
```
$$
JS_k(S, \hat{S}) =
\frac{1}{k}
\sum_{j=1}^{k}
\mathbf{1}\big(m_j(S) = m_j(\hat{S})\big)
$$