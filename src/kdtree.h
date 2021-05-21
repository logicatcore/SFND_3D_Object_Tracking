/* \author Aaron Brown */
// Quiz on implementing kd tree

// Structure to represent node of kd tree
template <typename PointT>
struct Node
{
	PointT point;
	int id;
	Node<PointT>* left;
	Node<PointT>* right;

	Node(std::vector<float> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}

	Node(PointT arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}

	~Node()
	{
		delete left;
		delete right;
	}
};

template <typename PointT>
struct KdTree
{
	Node<PointT> *root;

	KdTree(unsigned short d)
	: D(d), root(NULL)
	{}

	~KdTree()
	{
		delete root;
	}

	void place(Node<PointT> **n, std::vector<PointT> &point, int &id, uint depth) 
	{
		if(*n == NULL)
		{
			*n = new Node<PointT>(point, id);
		}
		else if(point[(int)(depth % D)] < (*n)->point[(int)(depth % D)])
		{
			place(&((*n)->left), point, id, ++depth);
		}
		else
		{
			place(&((*n)->right), point, id, ++depth);
		}
	}
	// overload to handle pcl point types, can handle PointXY and PointXYZ
	void place(Node<PointT> **n, PointT &point, int &id, uint depth) 
	{
		if(*n == NULL)
		{
			*n = new Node<PointT>(point, id);
		}
		else {
			switch (depth % D){
				case 0:
						if(point.x < (*n)->point.x)
						{
							place(&((*n)->left), point, id, ++depth);
						}
						else
						{
							place(&((*n)->right), point, id, ++depth);
						}
						break;
				case 1:
						if(point.y < (*n)->point.y)
						{
							place(&((*n)->left), point, id, ++depth);
						}
						else
						{
							place(&((*n)->right), point, id, ++depth);
						}
						break;
				case 2:
						if(point.z < (*n)->point.z)
						{
							place(&((*n)->left), point, id, ++depth);
						}
						else
						{
							place(&((*n)->right), point, id, ++depth);
						}
						break;
			}
		}
	}

	void insert(std::vector<PointT> point, int id)
	{
		// the function should create a new node and place correctly with in the root 
		place(&root, point, id, 0);
	}

  // overload to handle pcl point types, can handle PointXY and PointXYZ
	void insert(PointT point, int id)
	{
		// the function should create a new node and place correctly with in the root 
		place(&root, point, id, 0);
	}

	void digThrough(Node<PointT> **n, std::vector<PointT> *target, std::vector<int> *ids, float *tol, uint depth)
	{
		if(*n != NULL){
			if(abs((*target)[0] - (*n)->point[0]) <= *tol & abs((*target)[1] - (*n)->point[1]) <= *tol)
			{
				float dist = sqrt(pow((*target)[0] - (*n)->point[0], 2) + pow((*target)[1] - (*n)->point[1], 2));
				if (dist < *tol)
					(*ids).push_back((*n)->id);
			}

			uint cd = depth % D;
			if(((*target)[cd] - *tol) < (*n)->point[cd])
			{
				digThrough(&((*n)->left), target, ids, tol, ++depth);
			}
			if(((*target)[cd] + *tol) > (*n)->point[cd])
			{
				digThrough(&((*n)->right), target, ids, tol, ++depth);
			}
		}
	}

  // overload to handle pcl point types, can handle PointXY and PointXYZ
	void digThrough(Node<PointT> **n, PointT *target, std::vector<int> *ids, float *tol, uint depth)
	{
		if(*n != NULL){
			if(abs((*target).x - (*n)->point.x) <= *tol & abs((*target).y - (*n)->point.y) <= *tol & abs((*target).z - (*n)->point.z) <= *tol)
			{
				float dist = sqrt(pow((*target).x - (*n)->point.x, 2) + pow((*target).y - (*n)->point.y, 2) + pow((*target).z - (*n)->point.z, 2));
				if (dist < *tol)
					(*ids).push_back((*n)->id);
			}

			switch (depth % D){
				case 0:
							if(((*target).x - *tol) < (*n)->point.x)
							{
								digThrough(&((*n)->left), target, ids, tol, ++depth);
							}
							if(((*target).x + *tol) > (*n)->point.x)
							{
								digThrough(&((*n)->right), target, ids, tol, ++depth);
							}
							break;
				case 1:
							if(((*target).y - *tol) < (*n)->point.y)
							{
								digThrough(&((*n)->left), target, ids, tol, ++depth);
							}
							if(((*target).y + *tol) > (*n)->point.y)
							{
								digThrough(&((*n)->right), target, ids, tol, ++depth);
							}
							break;
				case 2:
							if(((*target).z - *tol) < (*n)->point.z)
							{
								digThrough(&((*n)->left), target, ids, tol, ++depth);
							}
							if(((*target).z + *tol) > (*n)->point.z)
							{
								digThrough(&((*n)->right), target, ids, tol, ++depth);
							}
							break;
			}
		}
	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(std::vector<PointT> target, float distanceTol)
	{
		std::vector<int> ids;
		digThrough(&root, &target, &ids, &distanceTol, 0);
		return ids;
	}

	// overload to handle pcl point types
	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(PointT target, float distanceTol)
	{
		std::vector<int> ids;
		digThrough(&root, &target, &ids, &distanceTol, 0);
		return ids;
	}

	private:
		const unsigned short D{0};
};




