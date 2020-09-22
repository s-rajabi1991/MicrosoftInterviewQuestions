using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.ComponentModel.Design;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Xml.Schema;

namespace MicrosoftInterviewQuestions
{
    public class Solution
    {
        public static void Main()
        {
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

        public IList<int> spiralOrder(int[][] matrix)
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

        //semi alternating-> without 3 consecutive
        public int longestSemiAlternatingSubstring(string s)
        {
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

            for(int i = 1; i < piles.Length; i++)
            {
                if (piles[i] != piles[i - 1])
                    differentNumbersCount++;

                result += differentNumbersCount;
            }

            return result;
        }

        public string dayOfWeekThatIsKDaysLater(string day, int k)
        {
            string[] days = new string[7] { "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun" };


            for(int i=0; i < days.Length; i++)
            {
                if (days[i] == day)
                    return days[(i + k)%7];
            }

            return "";
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
        public int maxInsertsToObtainStringWithout3ConsecutiveA(string s)
        {
            int numberOfAs = 0;

            int total = 0;

            for(int i = 0; i < s.Length; i++)
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

        //*****
        public int concatenatedStringLengthWithUniqueCharacters(IList<string> arr)
        {
            if (arr == null || arr.Count == 0)
                return 0;

            int result = 0;

            dfs(arr, "", 0, ref result);

            return result;
        }
        private void dfs(IList<string> arr , string path, int idx, ref int result)
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

        public int minSwapsToGroupRedBalls(string s)
        {
            List<int> redIndices = new List<int>();
            for (int i=0; i<s.Length;i++)
            {
                if (s[i] == 'R')
                    redIndices.Add(i);
            }

            int mid = redIndices.Count / 2;

            int result = 0;

            for (int i=0; i<redIndices.Count;i++)
            {
                int distanceToMid = Math.Abs(redIndices[i] - redIndices[mid]);
                int numberOfRsInBetween = Math.Abs(i - mid);
                result += distanceToMid - numberOfRsInBetween;
            }

            return result;

        }

        public int[] sumZero(int n)
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

            for (int i = start+1; i < result.Length; i+=2)
            {
                result[i] = n;
                result[i - 1] = -n;

                n++;
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
                    minDeletions = Math.Min(minDeletions, count_b_before+ count_a_after);
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

        //public List<List<int>> partitionArrayIntoNSubsetsWithBalancedSum(int[] A, int n)
        //{

        //}

        public bool canPartitionIntoKSubsetsWithEqualSum(int[] A, int k)
        {
            if (A == null || A.Length == 0 || k > A.Length)
                return false;

            int sum = A.Sum();

            if (sum % k != 0)
                return false;



        }

    }
}
