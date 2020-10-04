using NUnit.Framework;
using NUnit.Framework.Constraints;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FileSystem
{

    public abstract class Entry
    {
        protected string Name { get; set; }
        protected Directory Parent { get; set; }
        public Entry(string n, Directory parent)
        {
            Name = n;
            Parent = parent;
            CreatedDate = DateTime.Now;
            UpdatedDate = DateTime.Now;
            AccessedDate = DateTime.Now;
        }
        public string Path
        {
            get
            {
                if (Parent == null) return Name;

                else return Parent.Path + "/" + Name;
            }
        }
        protected DateTime CreatedDate { get; set; }
        protected DateTime UpdatedDate { get; set; }
        protected DateTime AccessedDate { get; set; }
        public void Rename(string n)
        {
            Name = n;
        }
        public void Delete()
        {
            Parent.DeleteEntry(this);
        }
        public abstract int Size();
    }

    public class Directory : Entry
    {
        private List<Entry> Contents { get; set; }
        public Directory(string n, Directory parent) : base(n, parent)
        {
            Contents = new List<Entry>();
        }
        public override int Size()
        {
            int size = 0;
            foreach (var content in Contents)
            {
                size += content.Size();
            }

            return size;
        }
        public void DeleteEntry(Entry entry)
        {
            Contents.Remove(entry);
        }
        public void AddEntry(Entry entry)
        {
            Contents.Add(entry);
        }
    }

    public class File : Entry
    {
        public File(string n, Directory parent, int size) : base(n, parent)
        {
            _Size = size;
        }
        public string Content { get; set; }
        private int _Size { get; set; }
        public override int Size()
        {
            return this._Size;
        }

    }

}
