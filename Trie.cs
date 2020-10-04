using System;
using System.Collections.Generic;
using System.Text;

namespace MicrosoftInterviewQuestions
{
    class Trie
    {
        public TrieNode Root { get; set; }
        public Trie()
        {
            Root = new TrieNode();
        }

        public void Insert(string word)
        {
            TrieNode node = Root;
            for(int i = 0; i < word.Length; i++)
            {
                if (!node.ContainsKey(word[i]))
                    node.Add(word[i]);

                node = node.Get(word[i]);
            }
            node.IsEnd = true;
        }
        public bool Search(string word)
        {
            TrieNode node = SearchPrefix(word);
            return node != null && node.IsEnd;
        }
        public bool StartsWith(string prefix)
        {
            TrieNode node = SearchPrefix(prefix);
            return node != null;
        }
        private TrieNode SearchPrefix(String word)
        {
            TrieNode node = Root;
            for (int i = 0; i < word.Length; i++)
            {
                if (node.ContainsKey(word[i]))
                    node = node.Get(word[i]);
                else
                    return null;

                node = node.Get(word[i]);
            }

            return node;
        }
    }

    class TrieNode
    {
        private TrieNode[] Links { get; set; }
        
        public bool IsEnd { get; set; }

        public TrieNode()
        {
            Links = new TrieNode[26];
        }

        public bool ContainsKey(char ch)
        {
            return Links[ch - 'a'] != null;
        }
        public TrieNode Get(char ch)
        {
            return Links[ch - 'a'];
        }
        public void Add(char ch)
        {
            Links[ch - 'a'] = new TrieNode();
        }
    }

    public class DictionaryNode
    {
        public DictionaryNode[] Children { get; set; }
        public int Ends { get; set; }

        public DictionaryNode()
        {
            Children = new DictionaryNode[26];
            Ends = 0;
        }
    }
    public class WordDictionary
    {
        public DictionaryNode Root { get; set; }
        /** Initialize your data structure here. */
        public WordDictionary()
        {
            Root = new DictionaryNode();
        }

        /** Adds a word into the data structure. */
        public void AddWord(string word)
        {
            DictionaryNode Node = Root;
            foreach(char c in word)
            {
                if (Node.Children[c - 'a'] == null)
                    Node.Children[c - 'a'] = new DictionaryNode();

                Node = Node.Children[c - 'a'];
            }
            Node.Ends++;
        }

        /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
        public bool Search(string word)
        {
            DictionaryNode node = this.Search(Root, word);
            return node!=null;
        }

        private DictionaryNode Search(DictionaryNode Node,string word)
        {
            for (int j = 0; j < word.Length; j++)
            {
                char c = word[j];
                if (c == '.')
                {
                    foreach (var child in Node.Children)
                    {
                        DictionaryNode node = Search(child, word.Substring(c + 1));
                        if (node != null)
                            return node;
                    }
                    return null;
                }
                else
                {
                    if (Node.Children[c - 'a'] != null)
                        Node = Node.Children[c - 'a'];
                    else
                        return null;
                }

            }

            return Node.Ends>0 ? Node : null;
        }
    }

}
