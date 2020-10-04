using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace MicrosoftInterviewQuestions
{
    public class MostFrequentQuestions
    {

        public int MinimumMoves(int[] arr)
        {
            #region Description
            //Given an integer array arr, in one move you can select a palindromic subarray arr[i], arr[i + 1], ..., arr[j] where i <= j, 
            //and remove that subarray from the given array.Note that after removing a subarray, the elements on the left and on the right of that subarray move to fill the gap left by the removal.
            //Return the minimum number of moves needed to remove all numbers from the array.
            #endregion

            if (arr == null || arr.Length == 0) return 0;

            if (arr.Length == 1) return 1;

            int[,] dp = new int[arr.Length, arr.Length];

            for (int k = 0; k < arr.Length; k++)
            {
                for (int i = 0; i < arr.Length - k; i++)
                {
                    int j = i + k;

                    if (i == j)
                        dp[i, j] = 1;

                    else if (j - i == 1)
                        dp[i, j] = arr[i] == arr[j] ? 1 : 2;

                    else
                    {
                        int min = Int32.MaxValue;

                        if (arr[i] == arr[j])
                        {
                            min = dp[i + 1, j - 1];
                        }

                        for (int mid = i; mid < j; mid++)
                        {
                            min = Math.Min(min, dp[i, mid] + dp[mid + 1, j]);
                        }

                        dp[i, j] = min;
                    }
                }

            }

            return dp[0, arr.Length - 1];
        }
        public int MaxDiff(int[] arr, int n)
        {
            #region Description
            //Maximum difference between two elements such that larger element appears after the smaller number
            #endregion

            int diff = Int32.MinValue;

            int maxRight = arr[n - 1];

            for (int i = n - 2; i >= 0; i--)
            {
                if (arr[i] > maxRight)
                {
                    maxRight = arr[i];
                }
                else
                {
                    diff = Math.Max(maxRight - arr[i], diff);
                }
            }
            return diff;
        }
        public IList<int> SpiralOrder(int[][] matrix)
        {
            IList<int> result = new List<int>();

            if (matrix.Length == 0 || matrix[0].Length == 0)
                return result;

            int rowBegin = 0;
            int rowEnd = matrix.Length - 1;
            int colBegin = 0;
            int colEnd = matrix[0].Length - 1;

            while (rowBegin <= rowEnd && colBegin <= colEnd)
            {
                for (int i = colBegin; i <= colEnd; i++)
                {
                    result.Add(matrix[rowBegin][i]);
                }
                rowBegin++;

                for (int i = rowBegin; i <= rowEnd; i++)
                {
                    result.Add(matrix[i][colEnd]);
                }
                colEnd--;

                if (rowBegin <= rowEnd && colBegin <= colEnd)
                {
                    for (int i = colEnd; i >= colBegin; i--)
                    {
                        result.Add(matrix[rowEnd][i]);
                    }
                    rowEnd--;

                    for (int i = rowEnd; i >= rowBegin; i--)
                    {
                        result.Add(matrix[i][colBegin]);
                    }
                    colBegin++;
                }

            }

            return result;
        }

        public string ReverseWords(string s)
        {
            List<string> allWords = new List<string>();

            StringBuilder currentWord = new StringBuilder();
            for (int i = 0; i < s.Length; i++)
            {
                if (!char.IsWhiteSpace(s[i]))
                {
                    currentWord.Append(s[i]);
                }
                else
                {
                    if (!string.IsNullOrEmpty(currentWord.ToString()))
                        allWords.Add(currentWord.ToString());

                    currentWord = new StringBuilder();
                }
            }

            if (!string.IsNullOrEmpty(currentWord.ToString()))
                allWords.Add(currentWord.ToString());


            int start = 0;
            int end = allWords.Count - 1;

            while (start < end)
            {
                string tmp = allWords[end];
                allWords[end] = allWords[start];
                allWords[start] = tmp;

                start++;
                end--;
            }

            return string.Join(' ', allWords);

        }

        public void ReverseWords(char[] s)//review
        {
            //Given an input string , reverse the string word by word. 

            ReverseString(s, 0, s.Length - 1);

            int start = 0;
            int end = 0;

            while (end < s.Length)
            {
                while (end < s.Length && s[end] != ' ')
                {
                    end++;
                }

                ReverseString(s, start, end - 1);

                start = end + 1;
                end = start;
            }

        }

        private void ReverseString(char[] s, int start, int end)
        {
            while (start < end)
            {
                char tmp = s[end];
                s[end] = s[start];
                s[start] = tmp;

                start++;
                end--;
            }
        }

        public bool IsValid(string s)
        {
            Dictionary<char, char> pairs = new Dictionary<char, char>();
            pairs.Add('(', ')');
            pairs.Add('{', '}');
            pairs.Add('[', ']');


            Stack<char> stack = new Stack<char>();

            for (int i = 0; i < s.Length; i++)
            {
                if (pairs.ContainsKey(s[i]))
                {
                    stack.Push(pairs[s[i]]);
                }
                else
                {
                    if (stack.Count == 0 || stack.Pop() != s[i])
                        return false;
                }
            }

            return stack.Count == 0;
        }
        public string LongestPalindrome(string s)//review
        {

            bool[,] dp = new bool[s.Length, s.Length];
            string longestPalindrome = "";

            for (int k = 0; k < s.Length; k++)
                for (int i = 0; i < s.Length - k; i++)
                {
                    int j = i + k;

                    if (i == j)
                        dp[i, j] = true;

                    else if (j - i == 1)
                        dp[i, j] = s[i] == s[j];

                    else
                    {
                        dp[i, j] = s[i] == s[j] && dp[i + 1, j - 1];
                    }

                    if (dp[i, j] && longestPalindrome.Length < j - i + 1)
                    {
                        longestPalindrome = s.Substring(i, j + 1);
                    }
                }

            return longestPalindrome;
        }
        public List<List<string>> GroupAnagrams(string[] strs)
        {
            Dictionary<string, List<string>> ans = new Dictionary<string, List<string>>();
            int[] count = new int[26];

            foreach (string s in strs)
            {
                Array.Fill(count, 0);

                foreach (char c in s)
                    count[c - 'a']++;


                String key = string.Join(',', count);

                if (!ans.ContainsKey(key))
                    ans.Add(key, new List<string> { s });
            }

            return ans.Select(i => i.Value).ToList();
        }// review what solutions there are (nklogk and nk)
        public void SetZeroes(int[][] matrix)//review efficient O(1) approach?
        {
            //approach 1: record rows and column if cell is 0 -> space: O(M+N)

            List<int> zeroRows = new List<int>();
            List<int> zeroCols = new List<int>();

            for (int i = 0; i < matrix.Length; i++)
                for (int j = 0; j < matrix[0].Length; j++)
                {
                    if (matrix[i][j] == 0)
                    {
                        zeroRows.Add(i);
                        zeroCols.Add(j);
                    }
                }


            foreach (int row in zeroRows)
                for (int j = 0; j < matrix[0].Length; j++)
                {
                    matrix[row][j] = 0;
                }



            foreach (int col in zeroCols)
                for (int i = 0; i < matrix.Length; i++)
                {
                    matrix[i][col] = 0;
                }


            //approach 2: flag first row and column
        }
        public void Rotate(int[,] matrix)// Review
        {
            //rotate matrix by 90 degree clockwise
            int n = matrix.GetLength(0);
            for (int layer = 0; layer < n / 2; layer++)
            {
                for (int i = layer; i < n - 1 - layer; i++)
                {

                    int top = matrix[layer, i];//save top
                    matrix[layer, i] = matrix[n - 1 - i, layer];//left ->top
                    matrix[n - 1 - i, layer] = matrix[n - 1 - layer, n - 1 - i];//bottom -> left
                    matrix[n - 1 - layer, n - 1 - i] = matrix[i, n - 1 - layer];//right -> bottom
                    matrix[i, n - 1 - layer] = top;//top -> right

                }
            }
        }
        public ListNode ReverseList(ListNode head)
        {
            #region Recursive Approach
            //if (head == null)
            //    return null;

            //if (head.next == null)
            //    return head;

            //ListNode rest = ReverseList(head.next);
            //head.next.next = head;
            //head.next = null;

            //return rest;
            #endregion

            #region Iterative Approach

            ListNode prev = null;
            ListNode curr = head;

            while (curr != null)
            {
                ListNode tmp = curr.next;
                curr.next = prev;
                prev = curr;
                curr = tmp;
            }

            return prev;

            #endregion


        } //Solve Again
        public bool HasCycle(ListNode head)//review two approaches
        {
            #region O(N) Memory
            //HashSet<ListNode> set = new HashSet<ListNode>();

            //while (head != null)
            //{
            //    if (set.Contains(head))
            //        return true;
            //    else
            //        set.Add(head);

            //    head = head.next;
            //}

            //return false;
            #endregion

            #region O(1) Memory
            if (head == null || head.next == null)
            {
                return false;
            }
            ListNode slow = head;
            ListNode fast = head.next;
            while (slow != fast)
            {
                if (fast == null || fast.next == null)
                {
                    return false;
                }
                slow = slow.next;
                fast = fast.next.next;
            }
            return true;
            #endregion

        }

        public ListNode AddTwoNumbers(ListNode l1, ListNode l2)
        {
            ListNode p = l1;
            ListNode q = l2;
            int c = 0;
            ListNode dummy = new ListNode(0);
            ListNode curr = dummy;

            while (p != null && q != null)
            {
                int sum = p.val + q.val + c;
                curr.next = new ListNode(sum % 10);

                c = sum / 10;
                curr = curr.next;
                p = p.next;
                q = q.next;
            }

            while (p != null)
            {
                int sum = p.val + c;
                curr.next = new ListNode(sum % 10);
                c = sum / 10;
                curr = curr.next;
                p = p.next;
            }

            while (q != null)
            {
                int sum = q.val + c;
                curr.next = new ListNode(sum % 10);
                c = sum / 10;
                curr = curr.next;
                q = q.next;
            }

            if (c > 0)
            {
                curr.next = new ListNode(c);
            }

            return dummy.next;
        }//simple but review

        public ListNode AddTwoNumbers2(ListNode l1, ListNode l2)
        {
            // If The most significant digit comes first 
            ListNode reversedL1 = ReverseList(l1);
            ListNode reversedL2 = ReverseList(l2);

            ListNode sum = AddTwoNumbers(reversedL1, reversedL2);

            return ReverseList(sum);
        }

        public ListNode MergeTwoLists(ListNode l1, ListNode l2)
        {
            ListNode p = l1;
            ListNode q = l2;

            ListNode dummy = new ListNode(0);
            ListNode curr = dummy;

            while (p != null && q != null)
            {
                if (p.val <= q.val)
                {
                    curr.next = new ListNode(p.val);
                    p = p.next;
                }
                else
                {
                    curr.next = new ListNode(q.val);
                    q = q.next;
                }

                curr = curr.next;
            }

            while (p != null)
            {
                curr.next = new ListNode(p.val);
                p = p.next;
                curr = curr.next;
            }

            while (q != null)
            {
                curr.next = new ListNode(q.val);
                q = q.next;
                curr = curr.next;
            }

            return dummy.next;
        }
        public ListNode MergeKLists(ListNode[] lists)
        {
            if (lists.Length == 0)
                return null;

            return MergeKListsHelper(lists, 0, lists.Length - 1);
        }
        private ListNode MergeKListsHelper(ListNode[] lists, int start, int end)
        {
            if (end == start)
                return lists[start];

            if (end == start + 1)
                return MergeTwoLists(lists[start], lists[end]);

            ListNode rest = MergeKListsHelper(lists, start + 1, end);
            ListNode list = MergeTwoLists(lists[start], rest);

            return list;
        }

        public ListNode GetIntersectionNode(ListNode headA, ListNode headB)//review different approaches
        {
            //1-> O(mn) brute force using two for
            //2-> O(m+n) hashtable
            //3-> O(n) time O(1) memory

            ListNode p = headA;
            ListNode q = headB;
            int lenA = 0;
            int lenB = 0;

            while (p != null)
            {
                p = p.next;
                lenA++;
            }

            while (q != null)
            {
                q = q.next;
                lenB++;
            }

            if (lenA > lenB)
            {
                while (lenA != lenB)
                {
                    headA = headA.next;
                    lenA--;
                }
            }
            else if (lenB > lenA)
            {
                while (lenB != lenA)
                {
                    headB = headB.next;
                    lenB--;
                }
            }


            while (headA != headB)
            {
                headA = headA.next;
                headB = headB.next;
            }

            return headB;
        }
        public Node CopyRandomList(Node head)// nice one! review!
        {

            #region O(N) time O(N) space (using dictionary to keep the old node --> new node mapping)
            //    //if (head == null)
            //    //{
            //    //    return null;
            //    //}

            //    //Dictionary<Node, Node> visited = new Dictionary<Node, Node>();

            //    //Node newHead = new Node(head.val);
            //    //visited.Add(head, newHead);

            //    //Node current = head;
            //    //Node newCurrent = newHead;

            //    //while (current != null)
            //    //{
            //    //    newCurrent.random = GetClonedNode(current.random,visited);
            //    //    newCurrent.next = GetClonedNode(current.next, visited);

            //    //    current = current.next;
            //    //    newCurrent = newCurrent.next;
            //    //}

            //    //return visited[head];
                #endregion

            //2-> O(1) space
            if (head == null)
                return null;

            Node current = head;

            while (current != null)
            {
                Node newNode = new Node(current.val);
                Node tmp = current.next;
                current.next = newNode;
                newNode.next = tmp;
                current = current.next.next;
            }

            current = head;

            while (current != null)
            {
                if (current.random != null)
                    current.next.random = current.random.next;
                current = current.next.next;
            }

            current = head;
            Node newHead = head.next;
            Node newCurrent = newHead;

            while (current != null)
            {
                current.next = current.next.next;

                if (newCurrent.next != null)
                    newCurrent.next = newCurrent.next.next;

                current = current.next;
                newCurrent = newCurrent.next;
            }

            return newHead;
        }
        public Node GetClonedNode(Node node, Dictionary<Node, Node> visited)
        {
            if (node != null)
            {
                if (visited.ContainsKey(node))
                {
                    return visited[node];
                }
                else
                {
                    visited.Add(node, new Node(node.val));
                    return visited[node];
                }
            }
            return null;
        }

        public bool IsValidBST(TreeNode root)
        {
            if (root == null)
                return true;

            return ValidateBST(root, null, null);
        }  //be careful with what is passed first

        private bool ValidateBST(TreeNode node, int? min, int? max)
        {
            bool valid = (min == null || node.val > min) && (max == null || node.val < max);

            if (node.left != null)
                valid = valid & ValidateBST(node.left, min, node.val);


            if (node.right != null)
                valid = valid & ValidateBST(node.right, node.val, max);


            return valid;

        }

        public IList<int> InorderTraversal(TreeNode root)
        {
            IList<int> result = new List<int>();

            if (root == null)
                return result;

            TraverseInOrder(root, result);

            return result;
        }

        private void TraverseInOrder(TreeNode node, IList<int> result)
        {
            if (node.left != null)
                TraverseInOrder(node.left, result);

            result.Add(node.val);

            if (node.right != null)
                TraverseInOrder(node.right, result);
        }

        public IList<IList<int>> LevelOrder(TreeNode root)
        {

            IList<IList<int>> result = new List<IList<int>>();

            if (root == null)
                return result;

            Queue<TreeNode> queue = new Queue<TreeNode>();
            queue.Enqueue(root);

            while (queue.Count > 0)
            {
                int count = queue.Count;
                IList<int> levelResult = new List<int>();

                while (count > 0)
                {
                    TreeNode node = queue.Dequeue();
                    levelResult.Add(node.val);

                    if (node.left != null)
                        queue.Enqueue(node.left);

                    if (node.right != null)
                        queue.Enqueue(node.right);

                    count--;
                }

                result.Add(levelResult);

            }

            return result;
        }

        public IList<IList<int>> ZigzagLevelOrder(TreeNode root)
        {
            IList<IList<int>> result = new List<IList<int>>();

            if (root == null)
                return result;

            Queue<TreeNode> q = new Queue<TreeNode>();
            q.Enqueue(root);

            bool leftToRight = true;

            while (q.Count > 0)
            {
                IList<int> levelResult = new List<int>();
                int count = q.Count;

                while (count > 0)
                {
                    TreeNode node = q.Dequeue();

                    if (leftToRight)
                        levelResult.Add(node.val);
                    else
                        levelResult.Insert(0, node.val);

                    if (node.left != null)
                        q.Enqueue(node.left);

                    if (node.right != null)
                        q.Enqueue(node.right);

                    count--;
                }

                result.Add(levelResult);
                leftToRight = !leftToRight;
            }

            return result;
        }

        public RNode Connect(RNode root) //perfect binary tree
        {
            #region O(1) space

            if (root == null || root.left == null)
                return root;

            RNode parent = root;

            while (parent.left != null)
            {
                RNode current = parent;

                while (current != null)
                {
                    current.left.next = current.right;

                    if (current.next != null)
                        current.right.next = current.next.left;

                    current = current.next;
                }


                parent = parent.left;
            }

            return root;

            #endregion
        }

        public RNode Connect2(RNode root) // if it is not perfect binary tree -> solve again
        {

            #region O(N) space
            //if (root == null)
            //    return root;

            //Queue<RNode> queue = new Queue<RNode>();
            //queue.Enqueue(root);


            //while (queue.Count > 0)
            //{
            //    int count = queue.Count;
            //    RNode prev = null;

            //    while (count > 0)
            //    {
            //        RNode node = queue.Dequeue();

            //        if (prev != null)
            //            prev.next = node;

            //        prev = node;

            //        if (node.left != null)
            //            queue.Enqueue(node.left);

            //        if (node.right != null)
            //            queue.Enqueue(node.right);


            //        count--;
            //    }


            //}

            //return root;

            #endregion

            if (root == null)
                return root;

            RNode head = root;

            while (head != null)
            {
                RNode curr = null;
                RNode nextLevelHead = null;

                while (head != null)
                {
                    if (head.left != null)
                    {
                        if (curr == null)
                        {
                            curr = head.left;
                            nextLevelHead = head.left;
                        }
                        else
                        {
                            curr.next = head.left;
                            curr = curr.next;
                        }
                    }

                    if (head.right != null)
                    {
                        if (curr == null)
                        {
                            curr = head.right;
                            nextLevelHead = head.right;
                        }
                        else
                        {
                            curr.next = head.right;
                            curr = curr.next;
                        }
                    }

                    head = head.next;

                }

                head = nextLevelHead;
            }

            return root;
        }

        public TreeNode BuildTree(int[] preorder, int[] inorder)
        {
            Dictionary<int, int> map = new Dictionary<int, int>();

            for (int i = 0; i < inorder.Length; i++)
            {
                map.Add(inorder[i], i);
            }

            return BuildTreeHelper(preorder, map, 0, preorder.Length - 1, 0, inorder.Length - 1);
        } //review

        private TreeNode BuildTreeHelper(int[] preorder, Dictionary<int, int> inMap, int preStart, int preEnd, int inStart, int inEnd)
        {
            if (preStart > preEnd || inStart > inEnd)
                return null;

            TreeNode root = new TreeNode(preorder[preStart]);

            int rootIdx = inMap[root.val];

            root.left = BuildTreeHelper(preorder, inMap, preStart + 1, preEnd, inStart, rootIdx - 1);
            root.right = BuildTreeHelper(preorder, inMap, preStart + rootIdx - inStart + 1, preEnd, rootIdx + 1, inEnd);

            return root;
        }

        public TreeNode LowestCommonAncestorInBST(TreeNode root, TreeNode p, TreeNode q)
        {
            #region O(n) space
            //if (root == null)
            //    return null;

            //else if (p.val < root.val && q.val < root.val)
            //    return LowestCommonAncestor(root.left, p, q);

            //else if (p.val > root.val && q.val > root.val)
            //    return LowestCommonAncestor(root.right, p, q);
            //else
            //    return root;
            #endregion

            #region O(1) space

            TreeNode curr = root;

            while (curr != null)
            {
                if (p.val < curr.val && q.val < curr.val)
                    curr = curr.left;

                else if (p.val > curr.val && q.val > curr.val)
                    curr = curr.right;

                else return curr;
            }

            return null;

            #endregion
        } //easy but review

        public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) //review
        {
            if (root == null || root == p || root == q)
                return root;


            TreeNode left = LowestCommonAncestor(root.left, p, q);
            TreeNode right = LowestCommonAncestor(root.right, p, q);

            return left == null ? right : right == null ? left : root;

        }

        public GNode CloneGraph(GNode node) //review
        {
            if (node == null)
            {
                return node;
            }

            Dictionary<int, GNode> visited = new Dictionary<int, GNode>();

            Queue<GNode> queue = new Queue<GNode>();
            queue.Enqueue(node);

            GNode newNode = new GNode(node.val, new List<GNode>());
            visited.Add(node.GetHashCode(), newNode);

            while (queue.Count > 0)
            {
                GNode curr = queue.Dequeue();

                foreach (GNode n in curr.neighbors)
                {
                    if (!visited.ContainsKey(n.GetHashCode()))
                    {
                        queue.Enqueue(n);
                        visited.Add(n.GetHashCode(), new GNode(n.val, new List<GNode>()));
                    }

                    visited[n.GetHashCode()].neighbors.Add(visited[n.GetHashCode()]);
                }

            }

            return visited[node.GetHashCode()];
        }

        public IList<string> LetterCombinations(string digits)
        {
            Dictionary<char, List<char>> mapping = new Dictionary<char, List<char>>();
            mapping.Add('2', new List<char> { 'a', 'b', 'c' });
            mapping.Add('3', new List<char> { 'd', 'e', 'f' });
            mapping.Add('4', new List<char> { 'g', 'h', 'i' });
            mapping.Add('5', new List<char> { 'j', 'k', 'l' });
            mapping.Add('6', new List<char> { 'm', 'n', 'o' });
            mapping.Add('7', new List<char> { 'p', 'q', 'r', 's' });
            mapping.Add('8', new List<char> { 't', 'u', 'v' });
            mapping.Add('9', new List<char> { 'w', 'x', 'y', 'z' });

            IList<string> result = new List<string>();

            if (string.IsNullOrEmpty(digits))
                return result;

            LetterCombinationsHelper(digits, 0, result, mapping, "");

            return result;
        } //review

        private void LetterCombinationsHelper(string digits, int index, IList<string> combinations, Dictionary<char, List<char>> mapping, string s)
        {

            if (index >= digits.Length)
            {
                combinations.Add(s);
                return;
            }

            List<char> chars = mapping[digits[index]];
            for (int j = 0; j < chars.Count; j++)
            {
                LetterCombinationsHelper(digits, index + 1, combinations, mapping, s + chars[j]);
            }

        }

        public IList<string> FindWords(char[][] board, string[] words)//review for sure!
        {
            //Approach 1 -> DFS + HM O(NM*NM* Count Of Words)
            //Approach 2 DFS + Trie!

            IList<string> result = new List<string>();
            Dictionary<char, List<int[]>> map = new Dictionary<char, List<int[]>>();

            for (int i = 0; i < board.Length; i++)
            {
                for (int j = 0; j < board[0].Length; j++)
                {
                    if (!map.ContainsKey(board[i][j]))
                        map.Add(board[i][j], new List<int[]>());

                    map[board[i][j]].Add(new int[] { i, j });
                }
            }

            foreach (string word in words)
            {
                if (!map.ContainsKey(word[0]))
                    continue;

                List<int[]> startingPoints = map[word[0]];
                for (int i = 0; i < startingPoints.Count; i++)
                {
                    if (FindWord(board, startingPoints[i][0], startingPoints[i][1], word, 0))
                    {
                        result.Add(word);
                        break;
                    }
                }
            }


            return result;
        }

        private bool FindWord(char[][] board, int i, int j, string word, int index)
        {
            if (index == word.Length)
                return true;

            if (i < 0 || i == board.Length || j < 0 || j == board[0].Length || board[i][j] != word[index])
                return false;

            board[i][j] = '#';

            int[] rowOffset = { 0, 0, 1, -1 };
            int[] colOffset = { 1, -1, 0, 0 };
            bool success = false;
            for (int k = 0; k < rowOffset.Length; k++)
            {
                success = FindWord(board, i + rowOffset[k], j + colOffset[k], word, index + 1);
                if (success)
                    break;
            }

            board[i][j] = word[index];
            return success;
        }

        public int RemoveDuplicates(int[] nums)
        {
            if (nums == null || nums.Length == 0 || nums.Length == 1)
                return 0;

            int u = 0;

            for (int i = 1; i < nums.Length; i++)
            {
                if (nums[i] != nums[i - 1])
                {
                    nums[u + 1] = nums[i];
                    u++;
                }
            }

            return u + 1;
        }//review

        public void Merge(int[] nums1, int m, int[] nums2, int n)//review
        {
            if (nums1.Length == 0 || nums2.Length == 0)
                return;

            int p = m - 1;
            int q = n - 1;

            int l = m + n - 1;

            while (p >= 0 && q >= 0)
            {
                nums1[l--] = nums1[p] > nums2[q] ? nums1[p--] : nums2[q--];
            }

            while (q >= 0)
            {
                nums1[l--] = nums2[q--];
            }


        }

        public int MaxProfit(int[] prices)//review
        {
            if (prices == null || prices.Length == 0) return 0;

            int n = prices.Length;

            int maxProfit = 0;
            int minPrice = prices[0];
            for (int i = 0; i < n; i++)
            {
                if (prices[i] < minPrice)
                    minPrice = prices[i];
                else
                    maxProfit = Math.Max(maxProfit, prices[i]-minPrice);

            }
            return maxProfit;
        }

        public int MaxSubArray(int[] nums)//review
        {

            int maxSum = nums[0];
            for (int i = 1; i < nums.Length; i++)
            {
                if (nums[i - 1] > 0)
                    nums[i] += nums[i - 1];

                maxSum = Math.Max(nums[i], maxSum);

            }

            return maxSum;
        }
        public int LengthOfLIS(int[] nums)//review
        {
            if (nums == null || nums.Length==0) return 0;
            //approach 1: Recursive O(2^n)
            //return LengthOfLISHelper(nums, Int32.MinValue, 0);

            //approach 2: DP O(n^2)
            int[] dp = new int[nums.Length];
            Array.Fill(dp, 1);

            for (int i = 1; i < nums.Length; i++) {

                for (int j = i - 1; j >= 0; j--)
                {
                    if (nums[i] > nums[j])
                        dp[i] = Math.Max(dp[i], 1 + dp[j]);
                }

            }

            return dp.Max();
        }

        private int LengthOfLISHelper(int[] nums,int prev, int idx)
        {
            if (idx == nums.Length)
                return 0;

            int taken = 0;
            if (nums[idx] > prev)
                taken = 1 + LengthOfLISHelper(nums, nums[idx], idx + 1);

            int nottaken = LengthOfLISHelper(nums, prev, idx + 1);
            return Math.Max(taken, nottaken);

        }

        public void SortColors(int[] nums)//review O(1) in one pass
        {
            int low = 0;
            int high = nums.Length - 1;

            int mid = 0;

            while (mid <= high)
            {
                if (nums[mid] == 0)
                {
                    int tmp = nums[low];
                    nums[low] = nums[mid];
                    nums[mid] = tmp;

                    low++;
                    mid++;
                }
                else if (nums[mid] == 1)
                {
                    mid++;
                }
                else
                {
                    int tmp = nums[high];
                    nums[high] = nums[mid];
                    nums[mid] = tmp;

                    high--;

                }
            }

        }

        public int FindMin(int[] nums) //review logn solution, what if there are duplicates?
        {
            if (nums == null || nums.Length == 0)
                return 0;
            int l = 0;
            int r = nums.Length - 1;

            while (l < r)
            {

                int mid = (l + r) / 2;

                if (nums[mid] > nums[r])
                {
                    l = mid + 1;
                }
                else
                {
                    r = mid;
                }

            }

            return nums[l];
        }

        public int SearchRotatedSortedArray(int[] nums, int target)
        {
            int l = 0;
            int r = nums.Length - 1;

            while (l < r)
            {
                int m = (l + r) / 2;

                if (nums[m] > nums[r])
                    l = m + 1;

                else
                    r = m;
            }

            if (target == nums[l])
                return l;

            r = nums.Length - 1;

            if (target <= nums[r])
                return BinarySearch(nums, l, r, target);

            else
                return BinarySearch(nums, 0, l-1 , target);
        }

        private int BinarySearch(int[] nums, int l, int r, int target)
        {
            if (l > r)
                return -1;

            int mid = (l + r) / 2;

            if (nums[mid] == target)
                return mid;

            if (target< nums[mid])
                return BinarySearch(nums, mid + 1, r, target);

            else
                return BinarySearch(nums, l, mid - 1, target);

        }
        public bool SearchMatrix(int[][] matrix, int target)
        {
            if (matrix == null || matrix.Length == 0 || matrix[0].Length == 0)
                return false;

            int m = matrix.Length;
            int n = matrix[0].Length;

            int l = 0;
            int r = m * n;

            while (l < r)
            {
                int mid = (l + r) / 2;

                int row = mid / n;
                int col = mid % n;

                if (target == matrix[row][col])
                    return true;
                else if (target < matrix[row][col])
                    r = mid;
                else
                    l = mid + 1;

            }

            return false;

        }

        public bool SearchMatrix2(int[,] matrix, int target)
        {
            if (matrix == null || matrix.GetLength(0) == 0 || matrix.GetLength(1) == 0)
                return false;

            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);

            int row = m - 1;
            int col = 0;

            while (row >= 0 && col < n)
            {
                int curr = matrix[row, col];

                if (target < curr)
                    row--;
                else if (target > curr)
                    col++;
                else
                    return true;

            }

            return false;

        }

        public int SingleNumber(int[] nums)//review approach with space O(1)
        {
            int a = 0;
            foreach (int i in nums)
            {
                a ^= i;
            }
            return a;

        }
        public int MyAtoi(string str)
        {
            int i = 0;
            while (i<str.Length && str[i] == ' ')
                i++;

            bool isNegative = i < str.Length && str[i] == '-';

            i++;


            int result = 0;
            while (i < str.Length && char.IsDigit(str[i]))
            {
                if (result > Int32.MaxValue / 10 ||  (result == Int32.MaxValue / 10 && str[i] - '0' > Int32.MaxValue % 10))
                {
                    return !isNegative ? Int32.MaxValue : Int32.MinValue;
                }
                result = result * 10 + (str[i] - '0');
            }

            return !isNegative ? -result : result;

        }






    }
    public class GNode
    {
        public int val;
        public IList<GNode> neighbors;

        public GNode()
        {
            val = 0;
            neighbors = new List<GNode>();
        }

        public GNode(int _val)
        {
            val = _val;
            neighbors = new List<GNode>();
        }

        public GNode(int _val, List<GNode> _neighbors)
        {
            val = _val;
            neighbors = _neighbors;
        }
    }
    public class RNode
    {
        public int val;
        public RNode left;
        public RNode right;
        public RNode next;

        public RNode() { }

        public RNode(int _val)
        {
            val = _val;
        }

        public RNode(int _val, RNode _left, RNode _right, RNode _next)
        {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }
    public class Node
    {
        public int val;
        public Node next;
        public Node random;

        public Node(int _val)
        {
            val = _val;
            next = null;
            random = null;
        }
    }
    public class ListNode
    {
        public int val;
        public ListNode next;
        public ListNode(int val = 0, ListNode next = null)
        {
            this.val = val;
            this.next = next;
        }
    }

}
