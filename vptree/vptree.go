// vptree.go, jpad 2016

/*
Package vptree implements a 'vantage point tree' for exact nearest neighbor
searches. The dataset consists of n d-dimensional points in a metric space.
Points with identical positions are treated as unique entities. Any datatype
that implements the Point interface can be handled.

See http://web.cs.iastate.edu/~honavar/nndatastructures.pdf :
"Data Structures and Algorithms for Nearest Neighbor Search
in General Metric Spaces"
by Peter N. Yianilos

TODO: serialize/deserialize
TODO: updates (insert/delete element)
*/
package vptree

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
)

//// Point /////////////////////////////////////////////////////////////////////

// Point in a metric space of arbitrary dimensions.
// The distance metric must fulfil the following requirements:
//
//	d(x, y) >= 0
//	d(x, y) = 0 if and only if x == y
//	d(x, y) = d(y, x)
//	d(x, z) <= d(x, y) + d(y, z) (triangle inequality)
type Point interface {
	// Distance returns the distance between the receiver and p.
	Distance(p Point) float32
}

//// VPTree ////////////////////////////////////////////////////////////////////

// VPTree represents a vantage point tree.
type VPTree struct {
	root *node

	seed uint64
	rnd  *rand.Rand
}

// NewVPTree returns an initialized tree holding the given points. The search
// performance of a tree varies with the queries used and the vantage point
// layout, but no heuristics are known to create an ideally efficient vp-tree.
// Therefore the vantage points are chosen randomly, and the random number
// generator seed can be specified, to be able to try multiple variants and
// pick the fastest.
func NewVPTree(points []Point, seed uint64) *VPTree {
	// points will be reordered, make a working copy
	_points := make([]Point, len(points))
	copy(_points, points)

	tree := &VPTree{
		seed: seed,
		rnd:  rand.New(rand.NewSource(int64(seed))),
	}
	tree.root = tree.build(_points)
	return tree
}

// String returns a textual rendering of the tree.
func (t *VPTree) String() string {
	var print func(*node, uint, *[]byte)
	print = func(n *node, depth uint, txt *[]byte) {
		if n == nil {
			return
		}
		line := fmt.Sprintf("%s%s boundary %0.3f, depth %d\n",
			indent(depth), n.p, n.boundary, depth)
		*txt = append(*txt, line...)

		// process child nodes
		print(n.inside, depth+1, txt)
		print(n.outside, depth+1, txt)
	}

	text := []byte{}
	print(t.root, 0, &text)
	return string(text)
}

// NN returns the nearest neighbor to the query point, together with its
// distance.
func (t *VPTree) NN(query Point) (Point, float32) {
	τ := float32(math.MaxFloat32)
	min := _Item{}
	t.nn(t.root, &τ, query, &min)

	return min.p, min.d
}

// KNN returns the k nearest neighbors to the query point, together with their
// respective distances. The results are sorted from nearest to farthest. If
// k > total number of samples in the tree, k is limited to the total number
// of samples. When there is a tie for the farthest of the nearest neighbors,
// k won't increase to always return all of them.
func (t *VPTree) KNN(query Point, k int) ([]Point, []float32) {
	if k < 1 || t.root == nil {
		return []Point{}, []float32{}
	} else if k == 1 {
		nn, dist := t.NN(query)
		return []Point{nn}, []float32{dist}
	}

	τ := float32(math.MaxFloat32)
	pq := make(priorityQueue, 0, k) // priority queue
	t.kNN(t.root, &τ, query, k, &pq)

	return extractReverse(pq)
}

// RangeNN returns the neighbors with a distance <= r to the query point,
// together with their respective distances. The results, if any, are sorted
// from nearest to farthest.
func (t *VPTree) RangeNN(query Point, r float32) ([]Point, []float32) {
	if r < 0.0 {
		return []Point{}, []float32{}
	}
	τ := r
	pq := make(priorityQueue, 0, 8) // priority queue
	t.rangeNN(t.root, &τ, query, &pq)

	return extractReverse(pq)
}

func (t *VPTree) Info() Info {
	info := Info{}

	var getInfo func(*node, uint, *Info)
	getInfo = func(n *node, depth uint, info *Info) {
		if n == nil {
			return
		}
		info.NNodes++
		if n.inside == nil && n.outside == nil {
			info.NLeaves++
		}
		if depth > info.MaxDepth {
			info.MaxDepth = depth
		}

		// process child nodes
		getInfo(n.inside, depth+1, info)
		getInfo(n.outside, depth+1, info)
	}

	getInfo(t.root, 0, &info)
	return info
}

//// unexported functionality //////////////////////////////////////////////////

// recursive balanced tree building
func (t *VPTree) build(points []Point) *node {
	if len(points) == 0 {
		return nil
	}
	n := &node{}

	// choose a vantage point
	if len(points) <= 2 {
		n.p = points[0]
		points = points[1:] // remove point 0
	} else {
		i := t.rnd.Intn(len(points))
		n.p = points[i]
		last := len(points) - 1
		points[i], points = points[last], points[:last] // remove point i
	}

	// partition the points into two equal-sized sets, one closer to the
	// node's center point than the median, and one farther away
	if len(points) > 0 {
		median := len(points) / 2
		selectKth(distancePoints{n.p, points}, median)

		n.boundary = points[median].Distance(n.p)
		n.inside = t.build(points[:median])
		n.outside = t.build(points[median:])
	}
	return n
}

// recursive nearest neighbor search
func (t *VPTree) nn(n *node, τ *float32, query Point, min *_Item) {
	if n == nil {
		return // empty tree: end recursion
	}

	d := n.p.Distance(query)
	if d < *τ {
		min.p = n.p
		min.d = d
		*τ = d
	}

	if n.inside == nil && n.outside == nil {
		return // leaf node: end recursion
	}

	if d < n.boundary {
		if d-*τ <= n.boundary {
			t.nn(n.inside, τ, query, min)
		}
		if d+*τ >= n.boundary {
			t.nn(n.outside, τ, query, min)
		}
	} else {
		if d+*τ >= n.boundary {
			t.nn(n.outside, τ, query, min)
		}
		if d-*τ <= n.boundary {
			t.nn(n.inside, τ, query, min)
		}
	}
}

// recursive nearest neighbors search
func (t *VPTree) kNN(
	n *node, τ *float32, query Point, k int, pq *priorityQueue,
) {
	if n == nil {
		return // empty tree: end recursion
	}

	d := n.p.Distance(query)
	if d < *τ {
		if pq.Len() == k {
			heap.Pop(pq)
		}
		heap.Push(pq, &_Item{n.p, d})
		if pq.Len() == k {
			*τ = pq.Top().(*_Item).d
		}
	}

	if n.inside == nil && n.outside == nil {
		return // leaf node: end recursion
	}

	if d < n.boundary {
		if d-*τ <= n.boundary {
			t.kNN(n.inside, τ, query, k, pq)
		}
		if d+*τ >= n.boundary {
			t.kNN(n.outside, τ, query, k, pq)
		}
	} else {
		if d+*τ >= n.boundary {
			t.kNN(n.outside, τ, query, k, pq)
		}
		if d-*τ <= n.boundary {
			t.kNN(n.inside, τ, query, k, pq)
		}
	}
}

// recursive range search
func (t *VPTree) rangeNN(
	n *node, τ *float32, query Point, pq *priorityQueue,
) {
	if n == nil {
		return // empty tree: end recursion
	}

	d := n.p.Distance(query)
	if d <= *τ {
		heap.Push(pq, &_Item{n.p, d})
	}

	if n.inside == nil && n.outside == nil {
		return // leaf node: end recursion
	}

	if d < n.boundary {
		if d-*τ <= n.boundary {
			t.rangeNN(n.inside, τ, query, pq)
		}
		if d+*τ >= n.boundary {
			t.rangeNN(n.outside, τ, query, pq)
		}
	} else {
		if d+*τ >= n.boundary {
			t.rangeNN(n.outside, τ, query, pq)
		}
		if d-*τ <= n.boundary {
			t.rangeNN(n.inside, τ, query, pq)
		}
	}
}

//// utilities /////////////////////////////////////////////////////////////////

func extractReverse(pq priorityQueue) ([]Point, []float32) {
	n := pq.Len()
	nn := make([]Point, n)
	distances := make([]float32, n)
	// extract neighbors and distances from priority queue
	for i := 0; i < n; i++ {
		hp := heap.Pop(&pq)
		nn[i] = hp.(*_Item).p
		distances[i] = hp.(*_Item).d
	}
	// reverse
	for i, j := 0, len(nn)-1; i < j; i, j = i+1, j-1 {
		nn[i], nn[j] = nn[j], nn[i]
		distances[i], distances[j] = distances[j], distances[i]
	}
	return nn, distances
}

//// unexported types //////////////////////////////////////////////////////////

// TODO: experiment with a node that stores inside lo/hi and outside lo/hi
type node struct {
	p        Point   // vantage point
	boundary float32 // distance to inside/outside boundary
	inside   *node   // subtree of points inside this node's hypersphere
	outside  *node   // subtree of points outside this node's hypersphere
}