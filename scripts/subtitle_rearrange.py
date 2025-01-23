import pandas as pd
import re
from rapidfuzz import process, fuzz

# 1. 读取数据文件和语句库文件
data_file = '/content/drive/MyDrive/Basic/short_character_anno.txt'
text_file = '/content/drive/MyDrive/Basic/text.csv'

# 读取数据文件
data_df = pd.read_csv(data_file, sep='|', names=['file_path', 'sentence'])

# 读取语句库文件
text_df = pd.read_csv(text_file, encoding='utf-8')

# 2. 使用正则表达式提取标签和中间内容
def extract_label_and_content(sentence):
    pattern = r'\[([A-Z]+)\](.*?)\[([A-Z]+)\]'
    match = re.search(pattern, sentence)
    if match:
        tag1, content, tag2 = match.groups()
        return tag1, content.strip(), tag2
    return None, None, None

# 应用提取函数
data_df[['tag1', 'content', 'tag2']] = data_df.apply(
    lambda row: pd.Series(extract_label_and_content(row['sentence'])), axis=1
)

# 3. 使用 rapidfuzz 匹配中间内容和语句库的 sentence 列
def fuzzy_match(content, text_df):
    # 使用 rapidfuzz 进行模糊匹配
    result = process.extractOne(content, text_df['sentence'], score_cutoff=72.5)
    if result:
        match, score, idx = result
        character = text_df.iloc[idx]['character']
        if len(match)/len(content)<0.71:
            with open("./log.txt","a") as f:
                f.write("error\n")
                f.write('content: '+content+'\nmatch: '+match+"\nscore: "+str(score)+"\n")
                return None, content
        return f"{character}", f"{match}"
    else:
        match, score, idx = process.extractOne(content, text_df['sentence'])
        with open("./log.txt","a") as f:
            f.write("error\n")
            f.write('content: '+content+'\nmatch: '+match+"\nscore: "+str(score)+"\n")
        return None, content  # 如果没有匹配项，返回 None

# 应用模糊匹配函数
data_df[['character', 'match']] = data_df.apply(
    lambda row: pd.Series(fuzzy_match(row['content'], text_df)), axis=1
)

# 4. 构造匹配结果列
data_df['match_result'] = data_df.apply(
    lambda row: f"[{row['tag1']}] {row['match']} [{row['tag2']}]" if pd.notnull(row['tag1']) else row['sentence'], axis=1
)

# 5. 选择需要的列并输出到文件
output_df = data_df[['file_path', 'character', 'match_result']]
output_df.to_csv('short_character.txt', sep='|', index=False, header=False)

