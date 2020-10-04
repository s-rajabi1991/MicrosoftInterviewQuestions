using System;
using System.Collections.Generic;
using System.Text;

namespace Serialization
{
    public class Codec
    {
        public string serialize(Node root)
        {
            StringBuilder sb = new StringBuilder();
            serializeHelper(root, sb);
            return sb.ToString();
        }
        private void serializeHelper(Node root, StringBuilder sb)
        {
            if (root == null)
                return;

            sb.Append(Convert.ToChar(root.val));

            foreach (Node child in root.children)
            {
                serializeHelper(child, sb);
            }

            sb.Append('_');
        }
        public Node deserialize(string data)
        {
            if (string.IsNullOrEmpty(data))
                return null;

            Queue<char> queue = new Queue<char>(data.ToCharArray());

            return deserializeHelper(queue);
        }
        private Node deserializeHelper(Queue<char> queue)
        {
            if (queue.Count == 0)
                return null;

            Node node = new Node(Convert.ToInt32(queue.Dequeue()), new List<Node>());

            while (queue.Peek() != '_')
            {
                node.children.Add(deserializeHelper(queue));
            }

            queue.Dequeue();//Discard -

            return node;
        }
    }

    public class Node
    {
        public int val;
        public IList<Node> children;

        public Node() { }

        public Node(int _val)
        {
            val = _val;
        }

        public Node(int _val, IList<Node> _children)
        {
            val = _val;
            children = _children;
        }
    }

}
