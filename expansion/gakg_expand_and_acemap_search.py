import pandas as pd
import os
import json
import requests
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from urllib.parse import quote

df = pd.read_parquet('gakg_subset.parquet')
#===================== 辅助函数 ========================#
#===================== URL 生成函数 ====================#
def generate_acemap_urls(results):
    """
    输入: advanced_geo_search 函数返回的字典
    输出: 包含分类 URL 的字典
    """
    base_url = "https://acemap.info/api/v1/work/search"
    url_report = {}

    for path_name, query_content in results.items():
        if isinstance(query_content, list):
            # 处理广度搜索中的多个 query
            urls = [f"{base_url}?keyword={quote(q)}&page=1&size=10" for q in query_content]
            url_report[path_name] = urls
        else:
            # 处理单条查询
            url = f"{base_url}?keyword={quote(query_content)}&page=1&size=10"
            url_report[path_name] = url
            
    return url_report
# ===================== 保存搜索结果函数 ====================#
def save_acemap_search_result(
    url,
    save_dir="acemap_results",
    timeout=10
):
    """
    给定 Acemap search API 链接，请求并将结果保存到本地（JSON）

    参数：
    - url: Acemap API 搜索链接
    - save_dir: 本地保存目录
    - timeout: 请求超时（秒）
    """

    os.makedirs(save_dir, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[ERROR] 请求失败: {e}")
        return None

    # ---------- 构造一个“可读文件名” ----------
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    keyword = params.get("keyword", ["unknown"])[0]
    page = params.get("page", ["1"])[0]
    size = params.get("size", ["10"])[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"acemap_{keyword}_p{page}_s{size}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)

    # ---------- 保存 ----------
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] 搜索结果已保存: {filepath}")
    return filepath

#===================== 搜索项计数器 =====================#

def get_search_result_count(url):
    """
    请求 Acemap URL 并返回该搜索项的结果总数
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    try:
        # 使用 verify=False 如果遇到证书问题，或者设置合理的 timeout
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # 根据 Acemap API 的常见结构提取总数
            # 优先级 1: 直接在根目录的 total (常见于分页 API)
            # 优先级 2: 在 results 列表的长度 (如果 API 只返回当前页)
            total = data.get('total') 
            
            if total is None:
                # 如果没有 total 字段，尝试获取结果列表的长度
                results = data.get('results', [])
                total = len(results)
                
            return total
        else:
            return "Error (Status: {})".format(response.status_code)
    except Exception as e:
        return f"Request Failed: {str(e)}"
    #===================== 辅助函数结束 =====================#


#===================== 多路径搜索策略 ====================#

def multi_path_broad_search_querys(keyword, df, top_k=5, use_relation=True):
    """
    keyword: 核心词
    df: GAKG 数据集
    top_k: 取前几个关联结果
    use_relation: 开关，True 则拼接三元组 (S-R-O)，False 则仅拼接核心词 (S-O)
    """
    # 1. 挖掘“果” (keyword 作为 Subject)
    eff_df = df[df['subject'].str.lower() == keyword.lower()]
    # 统计 (relation, object) 的组合频率
    top_effects = eff_df.groupby(['relation', 'object']).size().sort_values(ascending=False).head(top_k).index.tolist()
    
    # 2. 挖掘“因” (keyword 作为 Object)
    cau_df = df[df['object'].str.lower() == keyword.lower()]
    # 统计 (subject, relation) 的组合频率
    top_causes = cau_df.groupby(['subject', 'relation']).size().sort_values(ascending=False).head(top_k).index.tolist()
    
    querys = []

    # 3. 根据开关构建 Query
    # 处理“果”路
    for rel, obj in top_effects:
        if use_relation:
            querys.append(f"{keyword} {rel} {obj}") # 完整三元组语义
        else:
            querys.append(f"{keyword} {obj}")      # 仅核心词拼接
            
    # 处理“因”路
    for sub, rel in top_causes:
        if use_relation:
            querys.append(f"{sub} {rel} {keyword}") # 完整三元组语义
        else:
            querys.append(f"{sub} {keyword}")      # 仅核心词拼接

    # 去重（防止开关关闭后出现重复的词组）
    return list(dict.fromkeys(querys))

#===================== 实体挖掘函数 ====================#
import requests
from urllib.parse import quote

def get_top_entities(keyword, top_k=3):
    """
    通过关键词搜索论文，从返回的 results 中提取最相关的作者和机构。
    已适配 Acemap 真实返回结构：results -> authorships -> author/institutions
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    # 1. 发起请求：寻找 Top-K 篇论文
    # 使用你提供的标准搜索接口
    work_url = f"https://acemap.info/api/v1/work/search?keyword={quote(keyword)}&page=1&size={top_k}"
    
    authors_found = []
    orgs_found = []

    try:
        response = requests.get(work_url, headers=headers, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            # 关键修改 1：从 'results' 字段获取列表
            paper_list = res_json.get('results', [])
            
            for paper in paper_list:
                # 关键修改 2：从 'authorships' 字段提取信息
                authorships = paper.get('authorships', [])
                for authorship in authorships:
                    # 提取作者姓名
                    author_obj = authorship.get('author', {})
                    name = author_obj.get('display_name')
                    if name:
                        authors_found.append(name)
                    
                    # 关键修改 3：从 'institutions' 列表提取机构名
                    institutions = authorship.get('institutions', [])
                    for inst in institutions:
                        inst_name = inst.get('display_name')
                        if inst_name:
                            orgs_found.append(inst_name)
                
                """ # 关键修改 4：兜底方案 - 提取 host_organization_name
                primary_loc = paper.get('primary_location', {})
                if primary_loc:
                    host_org = primary_loc.get('source', {}).get('host_organization_name')
                    if host_org:
                        orgs_found.append(host_org)"""
        
        # 去重（保持顺序）
        unique_authors = list(dict.fromkeys(authors_found))[:top_k]
        unique_orgs = list(dict.fromkeys(orgs_found))[:top_k]

        # 2. 为这些实体生成专门的二级搜索 URL (作者和机构接口)
        author_urls = [f"https://acemap.info/api/v1/author/search?keyword={quote(a)}&page=1&size=5" for a in unique_authors]
        org_urls = [f"https://acemap.info/api/v1/institution/search?keyword={quote(o)}&page=1&size=5" for o in unique_orgs]

        return {
            "top_authors": unique_authors,
            "author_search_urls": author_urls,
            "top_institutions": unique_orgs,
            "institution_search_urls": org_urls
        }
    except Exception as e:
        print(f"实体挖掘失败: {e}")
        return {
            "top_authors": [], "author_search_urls": [],
            "top_institutions": [], "institution_search_urls": []
        }
    
"""def search_related_paperid(keyword, df, top_k=2):
    # 找到包含该关键词最频繁的 PaperID
    related_papers = df[(df['subject'].str.lower() == keyword.lower()) | 
                        (df['object'].str.lower() == keyword.lower())]
    
    top_paper_ids = related_papers['paperid'].value_counts().head(top_k).index.tolist()
    
    return top_paper_ids"""


def advanced_geo_search(keyword,  use_relation=True):
    print(f"--- 为关键词 [{keyword}] 构建多维搜索策略 ---\n")
    
    # --- 精准信息 ---
    precise_q = keyword # 直接使用关键词进行精准搜索
    
    # --- 广度拓展 ---
    # 寻果+索因 一共 2 * top_k 个扩展 query
    broad_qs = multi_path_broad_search_querys(keyword, df, top_k=5, use_relation=use_relation)
    
    # --- 深度拓展 ---
    # 目前精确信息去直接进行Acemap搜索，实际上返回的结果与地学知识基本不相干，这是因为
    # 目前precise keyword实际上并不“精准”，这是后期去优化的方法，那么目前用
    # 广度搜索结果去挖掘实体，作为临时代替
    entities_info = get_top_entities(broad_qs[0], top_k=5)

    # 根据精确信息搜索，找到相关的高频 PaperID
    #top_paper_ids = search_related_paperid(keyword, df)
    # 根据 PaperID，寻找最相关的作者或机构
    #entities_info = get_entities_by_paperids(top_paper_ids)

    querys = {
        "Precise": precise_q,
        "Broad/Causal": broad_qs,
    }

    
    return querys, entities_info


if __name__ == "__main__":

    queryset, entities_info = advanced_geo_search("Plate Tectonics",use_relation=True)

    keyword_urls = generate_acemap_urls(queryset)
    precise_url = keyword_urls['Precise']
    broad_urls = keyword_urls['Broad/Causal']
    author_urls = entities_info["author_search_urls"]
    institution_urls = entities_info["institution_search_urls"]

    #save_acemap_search_result(keyword_urls['Precise'])

    print("=====================关键词部分展示=====================\n")
    for k, v in queryset.items():
        print(f"{k}: {v}\n")

    print(entities_info["top_authors"])
    print(entities_info["top_institutions"])
    print("=====================URL部分展示=====================\n")

    print("---- 精准关键词搜索 URL ----\n")
    print(f"{precise_url}\n")
    print(f"搜索数:{get_search_result_count(precise_url)}\n")

    print("---- 扩展关键词搜索 URL ----\n")
    for url in broad_urls:
        print(f"{url}\n")
        print(f"搜索数:{get_search_result_count(url)}\n")

    print("---- 作者搜索 URL ----\n")
    for url in author_urls:
        print(f"{url}\n")
        print(f"搜索数:{get_search_result_count(url)}\n")

    print("---- 机构搜索 URL ----\n")
    for url in institution_urls:
        print(f"{url}\n")
        print(f"搜索数:{get_search_result_count(url)}\n")
    print("=====================================================\n")
    
