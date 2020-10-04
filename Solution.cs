using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel.Design;
using System.Globalization;
using System.Linq;
using System.Net.Mail;
using System.Reflection.Metadata.Ecma335;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading;
using System.Xml.Schema;

namespace MicrosoftInterviewQuestions
{
    public class Solution
    {
        public static void Main()
        {
        }

        #region Easy Ones
        public string longestSubstringWithout3ContiguousOccurrencesOfLetter(string s)
        {
            string longest = "";

            if (s == null || s.Length == 0)
                return longest;

            int start = 0;
            int end = 0;


            Dictionary<char, int> count = new Dictionary<char, int>();
            count.Add('a', 0);
            count.Add('b', 0);

            string lastValidSub = "";

            while (end < s.Length)
            {
                count[s[end]]++;
                count[s[end] == 'a' ? 'b' : 'a'] = 0;


                if (count[s[end]] > 2)
                {
                    longest = lastValidSub.Length > longest.Length ? lastValidSub : longest;


                    start = end - 1;
                    count[s[start]] = 1;
                    count[s[start] == 'a' ? 'b' : 'a'] = 0;

                    lastValidSub = s[start].ToString();
                }
                else
                {
                    lastValidSub += s[end];
                    end++;
                }
            }


            return string.IsNullOrEmpty(longest) ? lastValidSub : longest;

        }
        public string lexicographicallySmallestString(string s)
        {
            int greatestCharIndex = 0;
            char greatestChar = s[0];

            for (int i = 1; i < s.Length; i++)
            {
                if (s[i] > greatestChar)
                {
                    greatestCharIndex = i;
                    greatestChar = s[i];

                    if (s[i] == 'z')
                        break;
                }
            }


            StringBuilder strB = new StringBuilder(s);
            strB = strB.Remove(greatestCharIndex, 1);
            return strB.ToString();

        }
        public string stringWithout3IdenticalConsecutiveLetters(string s)
        {
            StringBuilder sb = new StringBuilder();

            int count = 1;
            sb.Append(s[0]);

            for (int i = 1; i < s.Length; i++)
            {
                if (s[i] == s[i - 1])
                {
                    count++;
                }
                else
                {
                    count = 1;
                }
                if (count < 3)
                    sb.Append(s[i]);
            }


            return sb.ToString();


        }
        public string dayOfWeekThatIsKDaysLater(string day, int k)
        {
            string[] days = new string[7] { "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun" };


            for (int i = 0; i < days.Length; i++)
            {
                if (days[i] == day)
                    return days[(i + k) % 7];
            }

            return "";
        }
        public int maxInsertsToObtainStringWithout3ConsecutiveA(string s)
        {
            int numberOfAs = 0;

            int total = 0;

            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == 'a')
                {
                    numberOfAs++;

                    if (numberOfAs >= 3)
                        return -1;
                }
                else
                {
                    total += 2 - numberOfAs;
                    numberOfAs = 0;
                }
            }

            total += 2 - numberOfAs;

            return total;
        }
        public int largestKSuchThatBothKAndMinusKExistInArray(int[] arr)
        {
            if (arr == null || arr.Length == 0 || arr.Length == 1)
                return 0;

            HashSet<int> set = new HashSet<int>();
            set.Add(arr[0]);

            int max = 0;
            for (int i = 1; i < arr.Length; i++)
            {
                if (set.Contains(arr[i] * -1))
                    max = Math.Max(Math.Abs(arr[i]), max);

                set.Add(arr[i]);
            }

            return max;
        }
        public int[] uniqueIntegersThatSumUpTo0(int n)
        {
            int[] result = new int[n];

            int start = 0;

            if (n % 2 == 1)
            {
                result[start] = 0;
                start++;
            }

            if (n == 1)
                return result;

            for (int i = start + 1; i < result.Length; i += 2)
            {
                result[i] = n;
                result[i - 1] = -n;

                n++;
            }

            return result;
        }
        public int[][] flipAndInvertImage(int[][] A)
        {
            if (A == null || A.Length == 0 || A[0].Length == 0)
                return A;


            for (int i = 0; i < A.Length; i++)
            {

                int end = A[i].Length - 1;
                int start = 0;

                while (start < end)
                {
                    int tmp = A[i][end];
                    A[i][end] = ~A[i][start];
                    A[i][start] = ~tmp;

                    end--;
                    start++;
                }

                if (start == end)
                    A[i][start] = ~A[i][start];

            }

            return A;

        }
        public bool leafSimilar(TreeNode root1, TreeNode root2)
        {
            if (root1 == null && root2 == null)
                return true;

            if (root1 == null || root2 == null)
                return false;

            List<int> leaves1 = new List<int>();
            List<int> leaves2 = new List<int>();

            getLeaves(leaves1, root1);
            getLeaves(leaves2, root2);

            if (leaves1.Count != leaves2.Count)
                return false;

            for (int i = 0; i < leaves1.Count; i++)
            {
                if (leaves1[i] != leaves2[i])
                    return false;
            }

            return true;
        }
        private void getLeaves(List<int> leaves, TreeNode root)
        {
            if (root.left == null && root.right == null)
            {
                leaves.Add(root.val);
                return;
            }

            if (root.left != null)
                getLeaves(leaves, root.left);

            if (root.right != null)
                getLeaves(leaves, root.right);

        }
        public int largestNumberXWhichOccurresXTimes(int[] A)
        {
            Dictionary<int, int> freq = new Dictionary<int, int>();

            for (int i = 0; i < A.Length; i++)
            {
                if (freq.ContainsKey(A[i]))
                    freq[A[i]]++;
                else
                    freq.Add(A[i], 1);
            }

            int max = 0;
            foreach (var pair in freq)
            {
                if (pair.Value == pair.Key)
                    max = Math.Max(max, pair.Key);
            }

            return max;
        }
        public string largestCharacter(string s)
        {
            HashSet<char> set = new HashSet<char>();
            set.Add(s[0]);

            char? max = null;

            for (int i = 1; i < s.Length; i++)
            {
                char lower = char.ToLower(s[i]);
                if (set.Contains(lower))
                {
                    max = max == null || max > lower ? lower : max;
                }
                else
                {
                    set.Add(lower);
                }
            }

            return max == null ? "NO" : max.Value.ToString();
        }
        public int widestPathWithoutTrees(int[] x, int[] y)
        {
            Array.Sort(x);

            int max = 0;

            for (int i = 1; i < x.Length; i++)
            {
                int distance = x[i] - x[i - 1];
                max = Math.Max(distance, max);
            }

            return max;
        }
        #endregion

        public int numbersWithEqualDigitsSum(int[] A)
        {
            //return maximum sum of two numbers whose digits add up to an equal sum

            Dictionary<int, int> digitSums = new Dictionary<int, int>();
            int max = -1;

            for (int i=0;i<A.Length;i++)
            {
                int currentSum = sumOfDigits(A[i]);
                
                if (digitSums.ContainsKey(currentSum))
                {
                    int last = digitSums[currentSum];
                    max = Math.Max(max, last + A[i]);
                    digitSums[currentSum] = Math.Max(last, A[i]);
                }
                else
                {
                    digitSums.Add(currentSum, A[i]);
                }
            }

            return max;
            
        }
        private int sumOfDigits(int number)
        {
            int sum = 0;
            while (number > 0)
            {
                sum += number % 10;
                number /= 10;
            }
            return sum;
        }
        public int minMovesForStringWithout3IdenticalConsecutiveLetters(string s)
        {
            int swaps = 0;
            for (int i = 0; i < s.Length; i++)
            {
                int count = 0;
                int j = i + 1;
                while (j < s.Length && s[i] == s[j])
                {
                    count++;
                    j++;
                }

                swaps += count / 3;

                i = j;
            }

            return swaps;

        }
        public int maxNetworkRank(int[] A, int[] B, int N)
        {
            int edgeNumbers = A.Length;
            int[] edgeCount = new int[N + 1];

            for (int i = 0; i < N + 1; i++)
            {
                edgeCount[A[i]]++;
                edgeCount[B[i]]++;
            }

            int maxRank = Int32.MinValue;
            for (int i = 0; i < edgeNumbers; i++)
            {
                int rank = edgeCount[A[i]] + edgeCount[B[i]] - 1;

                maxRank = Math.Max(maxRank, rank);

            }

            return maxRank;
        }
        public int minSwapsToMakePalindrome(string str)
        {
            StringBuilder strB = new StringBuilder(str);
            Dictionary<char, int> map = new Dictionary<char, int>();
            for (int i = 0; i < strB.Length; i++)
            {
                if (map.ContainsKey(strB[i]))
                    map.Remove(strB[i]);
                else
                    map.Add(strB[i], 1);
            }

            if (map.Count > 1)
                return -1;

            int start = 0;
            int end = strB.Length - 1;

            int total = 0;

            while (start < end)
            {
                if (strB[start] == strB[end])
                {
                    start++;
                    end--;
                }
                else
                {
                    int next = end;
                    while (next > start && strB[next] != strB[start])
                        next--;

                    if (next == start) //no matching found
                    {
                        swapWithNext(next, strB);
                        total++;
                    }
                    else
                    {
                        while (next < end)
                        {
                            swapWithNext(next, strB);
                            next++;
                            total++;
                        }
                    }

                }
            }

            return total;

        }
        private void swapWithNext(int index, StringBuilder strB)
        {
            char tmp = strB[index + 1];
            strB[index + 1] = strB[index];
            strB[index] = tmp;
        }          
        public int minDeletionsToMakeFrequencyOfEachLetterUnique(string s)
        {
            // counter of characters to delete
            int count = 0;
            // array of counters of occurrences for all possible characters
            Dictionary<char, int> freq = new Dictionary<char, int>();

            foreach (char c in s)
            {
                if (freq.ContainsKey(c))
                    freq[c]++;
                else
                    freq.Add(c, 1);
            }

            var heap = new C5.IntervalHeap<int>();
            heap.AddAll(freq.Select(i => i.Value));

            while (heap.Count > 0)
            {
                // take the biggest frequency of a character
                int most_frequent = heap.FindMax();

                heap.DeleteMax();

                if (heap.Count == 0) { return count; }
                // if this frequency is equal to the next one
                // and bigger than 1 decrease it to 1 and put
                // back to the queue
                if (most_frequent == heap.FindMax())
                {
                    if (most_frequent > 1)
                    {
                        heap.Add(most_frequent - 1);
                    }
                    count++;
                }
                // all frequencies which are bigger than
                // the next one are removed from the queue 
                // because they are unique
            }
            return count;
        }
        
        public int longestSemiAlternatingSubstring(string s)
        {
            //semi alternating-> without 3 consecutive letters
            if (s == null || s.Length == 0)
                return 0;

            int max = 1;

            int start = 0;
            int end = 1;

            int count = 1;

            while (end < s.Length)
            {
                if (s[end] == s[end - 1])
                    count++;
                else
                    count = 1;

                if (count == 3)
                {
                    start = end - 1;
                    count--;
                }

                max = Math.Max(max, end - start + 1);

                end++;

            }

            return max;

        }
        public int minStepstoMakePilesEqualHeight(int[] piles)
        {
            //record how many different numbers appeared before
            if (piles.Length == 0 || piles.Length == 1) return 0;

            int result = 0;
            int differentNumbersCount = 0;

            Array.Sort(piles);

            for (int i = 1; i < piles.Length; i++)
            {
                if (piles[i] != piles[i - 1])
                    differentNumbersCount++;

                result += differentNumbersCount;
            }

            return result;
        }
        public int maxPossibleValue(int a, int b, int n)
        {
            if (n == 2)
                return 0;

            int d = (n - 2 - Math.Abs(a - b) + 1) / 2;

            if (d == 0)
                return Math.Max(a, b) - 1;

            return Math.Max(a, b) + d;
        }
        public int MaxLengthOfConcatenatedStringWithUniqueCharacters(IList<string> arr)
        {
            if (arr == null || arr.Count == 0) return 0;

            HashSet<string> visited = new HashSet<string>();
            List<string> result = new List<string>();
            maxLengthHelper(arr, result, visited, "", 0);

            return result.Max(i => i.Length);
        }
        private void maxLengthHelper(IList<string> arr, List<string> result, HashSet<string> visited, string substring, int index)
        {
            if (!isValidSubstring(substring))
            {
                return;
            }

            if (index == arr.Count)
            {
                result.Add(substring);
                return;
            }

            for(int i = index; i < arr.Count; i++)
            {
                if (!visited.Contains(arr[i])) {
                    visited.Add(arr[i]);
                    maxLengthHelper(arr, result, visited, substring + arr[i], i+1);
                    visited.Remove(arr[i]);
                }
            }
        }        
        private bool isValidSubstring(string s)
        {
            HashSet<char> freq = new HashSet<char>();
            foreach(char c in s)
            {
                if (freq.Contains(c))
                    return false;

                freq.Add(c);
            }

            return true;
        } 
        public int concatenatedStringLengthWithUniqueCharacters(IList<string> arr)
        {
            if (arr == null || arr.Count == 0)
                return 0;

            int result = 0;

            dfs(arr, "", 0, ref result);

            return result;
        }
        private void dfs(IList<string> arr, string path, int idx, ref int result)
        {
            bool isUniqueChar = charactersAreUnique(path);

            if (!isUniqueChar)
                return;

            else
                result = Math.Max(path.Length, result);

            if (idx == arr.Count)
                return;

            for (int i = idx; i < arr.Count; i++)
            {
                dfs(arr, path + arr[i], i + 1, ref result);
            }
        }
        private bool charactersAreUnique(String s)
        {
            HashSet<char> set = new HashSet<char>();

            foreach (char c in s)
            {
                if (set.Contains(c))
                {
                    return false;
                }
                set.Add(c);
            }
            return true;
        }
        public int minSwapsToGroupRedBalls(string s)
        {
            List<int> redIndices = new List<int>();
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == 'R')
                    redIndices.Add(i);
            }

            int mid = redIndices.Count / 2;

            int result = 0;

            for (int i = 0; i < redIndices.Count; i++)
            {
                int distanceToMid = Math.Abs(redIndices[i] - redIndices[mid]);
                int numberOfRsInBetween = Math.Abs(i - mid);
                result += distanceToMid - numberOfRsInBetween;
            }

            return result;

        }
        public int minDeletionsToObtainStringInRightFormat(string s)
        {
            int count_a = 0;
            int count_b = 0;

            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == 'A')
                    count_a++;
                else
                    count_b++;
            }

            int minDeletions = Math.Min(count_a, count_b);

            int count_a_before = 0;
            int count_b_before = 0;

            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == 'A')
                {
                    count_a_before++;
                }
                else
                {
                    int count_a_after = count_a - count_a_before;
                    minDeletions = Math.Min(minDeletions, count_b_before + count_a_after);
                    count_b_before++;
                }
            }

            return minDeletions;
        }

        public int particleVelocity(List<int> a)
        {
            int totalPeriods = 0;
            for (int i = 0; i < a.Count; i++)
            {
                int count = 0;
                while (i + 2 < a.Count && a[i + 1] - a[i] == a[i + 2] - a[i + 1])
                {
                    count++;
                    totalPeriods += count;

                    i++;
                }
            }
            return totalPeriods < 1000000000 ? totalPeriods : -1;
        }

        public List<List<int>> canPartitionIntoKSubsetsWithEqualSum(int[] nums, int k)
        {
            List<List<int>> result = new List<List<int>>();

            if (nums == null || nums.Length == 0 || k > nums.Length)
                return result;

            int sum = nums.Sum();

            if (sum % k != 0)
                return result;

            bool[] visited = new bool[nums.Length];

            partitionIntoSubsets(nums, visited, 0, k, 0, sum / k, result, new List<int>());
            return result;
        }

        private bool partitionIntoSubsets(int[] nums, bool[] visited, int idx, int remaining, int sum, int target, List<List<int>> result, List<int> kResult)
        {
            if (remaining == 0)
            {
                return true;
            }

            if (sum == target)
            {
                result.Add(kResult);
                return partitionIntoSubsets(nums, visited, 0, remaining - 1, 0, target, result, new List<int>());
            }
            for (int i = idx; i < nums.Length; i++)
            {
                if (!visited[i] && nums[i] + sum <= target)
                {
                    visited[i] = true;
                    kResult.Add(nums[i]);

                    if (partitionIntoSubsets(nums, visited, i + 1, remaining, sum + nums[i], target, result, kResult))
                        return true;

                    kResult.Remove(nums[i]);
                    visited[i] = false;
                }
            }

            return false;
        }

        public int fairIndex(List<int> A, List<int> B)
        {
            int sumA = A.Sum();
            int sumB = B.Sum();

            int sumToThisPointA = A[0];
            int sumToThisPointB = B[0];

            int count = 0;
            for (int i = 1; i < A.Count; i++)
            {
                if (sumToThisPointA * 2 == sumA && sumToThisPointB * 2 == sumB && sumToThisPointA == sumToThisPointB)
                    count++;

                sumToThisPointA += A[i];
                sumToThisPointB += B[i];
            }
            return count;
        }
        public int minMeetingRooms(int[][] intervals)
        {
            if (intervals.Length == 0)
                return 0;

            intervals = intervals.OrderBy(i => i[0]).ToArray();

            SortedDictionary<int, int> heap = new SortedDictionary<int, int>(new MinComparer());
            heap.Add(intervals[0][1], 1);


            for (int i = 1; i < intervals.Length; i++)
            {
                var minFinish = heap.First();

                if (intervals[i][0] >= minFinish.Key)
                {
                    RemoveMin(heap);
                }

                Add(heap, intervals[i][1]);
            }



            return heap.Sum(i => i.Value);
        }
        private void RemoveMin(SortedDictionary<int, int> sd)
        {
            var first = sd.First();

            if (first.Value > 1)
                sd[first.Key]--;
            else
                sd.Remove(first.Key);
        }
        private void Add(SortedDictionary<int, int> sd, int val)
        {
            if (sd.ContainsKey(val))
                sd[val]++;
            else
                sd.Add(val, 1);
        }

        public int numTimesAllBlue(int[] light)
        {
            int moments = 0;
            int allOnBefore = 0;
            int[] on = new int[light.Length];

            for (int i = 0; i < light.Length; i++)
            {
                on[light[i] - 1] = 1;
                if (allOnBefore == light[i] - 1)
                {
                    while (allOnBefore < on.Length && on[allOnBefore] == 1)
                    {
                        allOnBefore++;
                    }
                    if (i + 1 == allOnBefore)
                    {
                        moments++;
                    }
                }
            }

            return moments;
        }

        public int maxChunksToSortArray(int[] A)
        {
            int n = A.Length;

            int[] right = new int[n];
            right[n - 1] = A[n - 1];

            for (int i = n - 2; i >= 0; i--)
            {
                right[i] = Math.Min(right[i + 1], A[i]);
            }

            int count = 0;
            int max = A[0];
            for (int i = 0; i < n; i++)
            {
                max = Math.Max(max, A[i]);
                if (i == n - 1 || max < right[i + 1])
                {
                    count++;
                }
            }
            return count;
        }
        public int largestMAlignedSubset(int[] A, int m)
        {
            if (A == null || A.Length == 0)
                return 0;

            Dictionary<int, int> remainders = new Dictionary<int, int>();

            foreach (int a in A)
            {
                int r = a < 0 ? (a % m) + m : a % m;

                if (remainders.ContainsKey(r))
                    remainders[r]++;
                else
                    remainders.Add(r, 1);

            }

            return remainders.Max(i => i.Value);
        }

        public int minCostToGetStringWithout2IdenticalConsecutiveLetters(string s, int[] C)
        {
            if (string.IsNullOrEmpty(s) || s.Length == 0) return 0;

            int total = 0;
            int last = C[0];
            for (int i = 1; i < s.Length; i++)
            {
                if (s[i] == s[i - 1])
                {
                    total += Math.Min(C[i], last);
                    last = Math.Max(C[i], last);
                }
                else
                {
                    last = C[i];
                }
            }

            return total;
        }

        public string riddle(string str)
        {
            StringBuilder sb = new StringBuilder(str);
            char[] letters = new char[] { 'a', 'b', 'c' };

            for (int i = 0; i < str.Length; i++)
            {
                if (str[i] == '?')
                {
                    foreach (var c in letters)
                    {
                        if ((i == 0 || str[i - 1] != c) && (i == str.Length - 1 || str[i + 1] != c))
                        {
                            sb[i] = c;
                            break;
                        }
                    }
                }
            }

            return sb.ToString();
        }

        public int countVisibleNodesInBinaryTree(TreeNode root)//or good nodes
        {
            if (root == null) return 0;

            List<int> result = new List<int>();

            countVisibleDfs(root, result, root.val);
            return result.Count;
        }

        private void countVisibleDfs(TreeNode node, List<int> values, int max)
        {
            if (node.val >= max)
            {
                values.Add(node.val);
                max = node.val;
            }

            if (node.left != null)
                countVisibleDfs(node.left, values, max);

            if (node.right != null)
                countVisibleDfs(node.right, values, max);

            return;
        }

        public int numberOfFractionsThatSumUpTo1(int[] X, int[] Y)
        {
            Dictionary<decimal, int> dict = new Dictionary<decimal, int>();
            int total = 0;
            for (int i = 0; i < X.Length; i++)
            {
                decimal d = (decimal)X[i] / (decimal)Y[i];

                if (dict.ContainsKey(1 - d))
                    total += dict[1 - d];

                if (dict.ContainsKey(d))
                    dict[d]++;
                else
                    dict.Add(d, 1);

            }

            return total;
        }

        public int possibleHoursVariations(int A, int B, int C, int D)
        {
            //return the count of how many variants are there to combine the four integers so its a valid hour
            // from 00:00 to 24:00

            int[] digits = new int[] { A, B, C, D };
            bool[] used = new bool[4];
            List<int> result = new List<int>();

            possibleHoursHelper(digits, used, result, 0, 0, 0);

            return result.Count;
        }

        private void possibleHoursHelper(int[] digits, bool[] used, List<int> result, int idx, int h, int m)
        {
            if (h > 23 || m > 59) return;

            if (idx == digits.Length - 1)
            {
                result.Add(h * 60 + m);
                return;
            }

            for (int i = 0; i < digits.Length; i++)
            {
                if (!used[digits[i]])
                {
                    used[i] = true;

                    if (idx == 0 || idx == 1)
                        possibleHoursHelper(digits, used, result, idx + 1, h * 10 + digits[i], m);

                    else
                        possibleHoursHelper(digits, used, result, idx + 1, h, m * 10 + digits[i]);

                    used[i] = false;
                }
            }
        }

        public bool canReach(int[] arr, int start)
        {
            if (arr == null || arr.Length == 0)
                return false;

            Queue<int> qu = new Queue<int>();
            bool[] visited = new bool[arr.Length];

            qu.Enqueue(start);
            visited[start] = true;

            while (qu.Count > 0)
            {
                int curr = qu.Dequeue();

                if (arr[curr] == 0)
                    return true;

                int[] neighbours = new int[] { curr + arr[curr], curr - arr[curr] };

                foreach (int n in neighbours)
                {
                    if (n >= 0 && n < arr.Length && !visited[n])
                    {
                        visited[n] = true;
                        qu.Enqueue(n);
                    }
                }

            }

            return false;
        }

        public int maxValueByInserting5(int n)
        {
            StringBuilder sb = new StringBuilder(n.ToString());

            if (char.IsDigit(sb[0])){

                for (int i = 0; i < sb.Length; i++)
                {
                    if (sb[i] < '5')
                        return Int32.Parse(sb.Insert(i, '5').ToString());
                }
                return Int32.Parse(sb.Append('5').ToString());
            }
            else
            {
                sb.Remove(0, 1);
                for (int i = sb.Length-1; i >=0; i--)
                {
                    if (sb[i] < '5')
                        return Int32.Parse(sb.Insert(i + 1, '5').ToString()) * -1;
                }
                return Int32.Parse(sb.Insert(0, '5').ToString())*-1;

            }
        }

        #region My Online AssessmentQuestions
        public int solve(int[] A)
        {
            if (A == null) return 0;

            Array.Sort(A);

            int first = 0;
            int last = A.Length - 1;

            if (A[first] >= 0 || A[last] < 0) return 0;

            while (first < last)
            {
                if (A[first] == -A[last])
                    return A[last];

                if (A[first] > -A[last])
                    last--;
                else
                    first++;
            }

            return 0;
        }
        public int solve2(string S, int[] C)
        {
            if (string.IsNullOrEmpty(S) || S.Length == 1)
                return 0;

            int totalCost = 0;
            int lastHighestCost = C[0];

            for (int i = 1; i < S.Length; i++)
            {
                if (S[i] == S[i - 1])
                {
                    totalCost += Math.Min(C[i], lastHighestCost);
                    lastHighestCost = Math.Max(C[i], lastHighestCost);
                }
                else
                {
                    lastHighestCost = C[i];
                }
            }

            return totalCost;

        }
        public int solve3(int[] X, int[] Y)
        {
            if (X == null || Y == null || X.Length == 0 || Y.Length == 0 || X.Length != Y.Length)
                return 0;

            int count = 0;
            Dictionary<decimal, int> dict = new Dictionary<decimal, int>();

            for (int i = 0; i < X.Length; i++)
            {
                decimal key = (decimal)X[i] / (decimal)Y[i];
                decimal otherKey = 1 - key;

                if (dict.ContainsKey(otherKey))
                {
                    count += dict[otherKey];
                }

                if (dict.ContainsKey(key))
                {
                    dict[key]++;
                }
                else
                {
                    dict.Add(key, 1);
                }
            }


            return count % 1000000007;
        }
        #endregion



    }
    public class MinComparer : IComparer<int>
    {
        public int Compare(int x, int y)
        {
            return x-y;
        }
    }


    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
        {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }


}
