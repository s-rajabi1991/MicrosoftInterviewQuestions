using System;
using System.Collections.Generic;
using System.Text;

namespace MicrosoftInterviewQuestions
{
    class LRUCache
    {
        private int Capacity { get; set; }
        private Dictionary<int, DNode> Map { get; set; }
        private DNode Head { get; set; }
        private DNode Tail { get; set; }
        public LRUCache(int capacity)
        {
            Capacity = capacity;
            Map = new Dictionary<int, DNode>();
            Head = new DNode();
            Tail = new DNode();
            Head.Next = Tail;
            Tail.Prev = Head;
        }

        public int Get(int key)
        {
            if (Map.ContainsKey(key))
            {
                DNode node = Map[key];
                RemoveNode(node);
                AddToFront(node);
                return node.Value;
            }
            return -1;
        }

        private void AddToFront(DNode node)
        {
            DNode next = Head.Next;
            node.Next = next;
            Head.Next = node;
            next.Prev = node;
            node.Prev = Head;
        }

        private void RemoveNode(DNode node)
        {
            node.Prev.Next = node.Next;
            node.Next.Prev = node.Prev;
        }
        private void Put(int key, int value)
        {
            if (Map.ContainsKey(key))
            {
                DNode node = Map[key];
                node.Value = value;
                RemoveNode(node);
                AddToFront(node);
            }
            else
            {
                if (Map.Count == Capacity)
                {
                    DNode last = Tail.Prev;
                    RemoveNode(last);
                    Map.Remove(last.Key);
                }

                DNode node = new DNode(key, value);
                AddToFront(node);

                Map.Add(key, node);

            }
        }
    }

    public class DNode
    {
        public DNode()
        {

        }
        public DNode(int key, int val)
        {
            Key = key;
            Value = val;
        }
        public int Value { get; set; }
        public int Key { get; set; }
        public DNode Next { get; set; }
        public DNode Prev { get; set; }
    }
}

