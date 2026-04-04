"""交互式问答启动脚本"""
import os
import sys
from config import Config
from reasoning_rag import ReasoningRAG
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🧠 REASONING RAG - Interactive Mode 🧠             ║
    ║                                                              ║
    ║  Ask any question and see the complete reasoning process!   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    index_path = './bioasq_index.pkl'

    # 检查索引是否存在
    if not os.path.exists(index_path):
        print(f"\n⚠️  Index file not found at {index_path}")
        print("Please run: python main.py --mode build")
        print("Or specify a different index path.\n")
        sys.exit(1)

    # 初始化配置和RAG系统
    print("\n🔧 Initializing Reasoning RAG system...")
    config = Config()
    rag_system = ReasoningRAG(config)

    # 加载索引
    print(f"📂 Loading index from {index_path}...")
    try:
        rag_system.load_index(index_path)
        print("✅ Index loaded successfully!\n")
    except Exception as e:
        print(f"\n❌ Error loading index: {e}\n")
        sys.exit(1)

    # 示例问题
    print("💡 Example questions you can ask:")
    print("   • What are the main symptoms of diabetes?")
    print("   • How does photosynthesis work in plants?")
    print("   • What is the relationship between DNA and proteins?")
    print("   • Explain the causes of climate change.")
    print("\n📝 Tips:")
    print("   • Ask complex questions to see multi-hop reasoning")
    print("   • Type 'quit' or 'exit' to stop")
    print("   • Press Ctrl+C to interrupt\n")

    question_count = 0

    while True:
        try:
            # 获取用户输入
            print(f"{'─'*70}")
            user_input = input("\n💭 Your Question: ").strip()

            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for using Reasoning RAG!\n")
                break

            # 检查空输入
            if not user_input:
                print("⚠️  Please enter a question.")
                continue

            question_count += 1
            print(f"\n{'═'*70}")
            print(f"🔍 Processing Query #{question_count}")
            print(f"{'═'*70}\n")

            # 处理问题
            result = rag_system.query(user_input, verbose=False)

            # 显示简洁结果
            print(f"\n📊 Analysis:")
            print(f"   Complexity: {result['analysis']['complexity_score']:.2f} ({'Complex' if result['analysis']['is_complex'] else 'Simple'})")

            if len(result['subqueries']) > 1:
                print(f"\n🧩 Sub-queries ({len(result['subqueries'])}):")
                for i, sq in enumerate(result['subqueries'], 1):
                    print(f"   {i}. {sq['subquery']}")

            print(f"\n📚 Retrieved {len(result['evidence']['evidence'])} pieces of evidence")
            print(f"   Average relevance: {result['retrieval']['retrieval_stats']['avg_similarity']:.3f}")

            print(f"\n{'─'*70}")
            print(f"💡 ANSWER (Confidence: {result['answer']['confidence']:.3f})")
            print(f"{'─'*70}\n")
            print(result['answer']['answer'])

            if result['answer']['sources']:
                print(f"\n📖 Top Sources:")
                for i, source in enumerate(result['answer']['sources'][:3], 1):
                    text = source['text'][:150] + "..." if len(source['text']) > 150 else source['text']
                    print(f"   [{i}] (score: {source['quality_score']:.2f}) {text}")

            print(f"\n{'═'*70}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!\n")
            break
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
            print("Please try another question.\n")

    # 显示会话统计
    if question_count > 0:
        print(f"\n{'═'*70}")
        print(f"📈 Session Statistics:")
        print(f"   Total Questions: {question_count}")
        print(f"{'═'*70}\n")

if __name__ == '__main__':
    main()