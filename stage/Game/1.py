def print_tree(node, indent=0):
    prefix = "  " * indent  # 使用空格缩进表示层级

    # 处理根节点
    if node["type"] == "root":
        print(f"{prefix}Root (depth: {node['depth']}, alpha: {node['alpha']}, beta: {node['beta']})")
    # 处理移动节点
    else:
        move = node["move"]
        from_pos = move["from"]
        to_pos = move["to"]
        piece = move["piece"]
        value = node["value"]
        pruned = node.get("pruned", False)  # 默认未剪枝为False
        print(f"{prefix}|-- Move: {piece} from {from_pos} to {to_pos}, value: {value}, pruned: {pruned}")

    # 递归处理子节点
    for child in node.get("children", []):
        print_tree(child, indent + 1)


# 示例用法
import json

with open("decision_tree.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
print_tree(json_data)