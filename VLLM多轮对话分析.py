from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
import torch
import os

class Phi3VisionChat:
    def __init__(self, model_id="microsoft/Phi-3-vision-128k-instruct", device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"正在加载模型... (使用设备: {device})")
        
        # 1. 加载并修改配置
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # 禁用 FlashAttention，我的笔记本上没有芯片，所以运行非常慢，在后面有服务器了之后可以接入以提高速度
        if hasattr(config, "attn_config"):
            config.attn_config["flash_attn"] = False
            config.attn_config["flash_rotary"] = False
        else:
            if hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"
            if hasattr(config, "use_flash_attention_2"):
                config.use_flash_attention_2 = False

        config.model_type = "phi3_v"

        # 因为笔记本的内存空间太小了，在笔记本上跑不通，看以后在服务器上用，但是又尝试了一遍之后又跑通了

        # 2. 加载模型和处理器
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32
        )
        print("模型加载完成!")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("处理器加载完成!")
        
        # 初始化对话状态
        self.conversation_history = []
        self.images = []
        
        # 设置生成参数
        self.generation_args = {
            "max_new_tokens": 500,  
            "temperature": 0.3,    # 稍高温度以获得更有创意的回答，但是这个我不太理解
            "do_sample": True,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
    
    def generate_response(self, user_input, new_image=None):
        """处理用户输入并生成助手响应"""
        try:
            # 添加用户消息到历史
            user_message = {"role": "user", "content": user_input}
            
            # 如果有新图片则处理
            if new_image:
                # 分配图片标记
                img_idx = len(self.images) + 1
                user_message["content"] = f"<|image_{img_idx}|>\n" + user_input
                self.images.append(new_image)
            
            self.conversation_history.append(user_message)
            
            # 生成完整提示
            prompt = self.processor.tokenizer.apply_chat_template(
                self.conversation_history,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 处理输入（包含所有历史图片）
            inputs = self.processor(
                text=prompt,
                images=self.images if self.images else None,
                return_tensors="pt"
            ).to(self.model.device)
            
            # 生成响应
            generate_ids = self.model.generate(
                **inputs,
                **self.generation_args
            )
            
            # 解码响应
            full_response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # 提取助手回复
            if "<|assistant|>" in full_response:
                response = full_response.split("<|assistant|>")[-1].strip()
            elif "assistant" in full_response:
                # 处理可能的分割情况
                parts = full_response.split("assistant")
                response = parts[-1].strip() if len(parts) > 1 else full_response
            else:
                response = full_response
            
            # 添加助手回复到历史
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            return f"生成响应时出错: {str(e)}"
    
    def reset_history(self):
        """重置对话历史"""
        self.conversation_history = []
        self.images = []
        return "对话历史已重置!"
    
    def show_history(self):
        """显示对话历史"""
        if not self.conversation_history:
            return "对话历史为空!"
        
        history_str = "\n对话历史:\n" + "-"*40
        for i, msg in enumerate(self.conversation_history):
            role = msg["role"]
            content = msg["content"]
            
            # 截断长内容以便显示
            if len(content) > 100:
                content = content[:100] + "..."
            
            history_str += f"\n[{i+1}] {role.capitalize()}: {content}"
        history_str += "\n" + "-"*40
        return history_str

def interactive_chat():
    """启动交互式聊天界面"""
    # 创建聊天实例
    chat = Phi3VisionChat()
    
    print("\n" + "="*60)
    print("Phi-3 Vision 聊天助手已启动!")
    print("输入 'exit' 退出, 'reset' 重置对话, 'history' 查看历史")
    print("要添加图片, 输入: /image <图片路径>")
    print("="*60 + "\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n您: ").strip()
            
            # 退出命令
            if user_input.lower() in ["exit", "quit"]:
                print("退出聊天...")
                break
                
            # 重置对话
            if user_input.lower() == "reset":
                print(chat.reset_history())
                continue
                
            # 查看历史
            if user_input.lower() == "history":
                print(chat.show_history())
                continue
                
            # 处理图片命令
            new_image = None
            if user_input.startswith("/image "):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print("错误: 请提供图片路径. 例如: /image path/to/image.jpg")
                    continue
                
                image_path = parts[1].strip()
                try:
                    if not os.path.exists(image_path):
                        print(f"错误: 文件不存在 - {image_path}")
                        continue
                    
                    new_image = Image.open(image_path)
                    print(f"已加载图片: {image_path}")
                    
                    # 获取图片相关的问题
                    user_input = input("请提供关于此图片的问题: ").strip()
                    if not user_input:
                        print("错误: 需要提供问题!")
                        continue
                except Exception as e:
                    print(f"加载图片时出错: {str(e)}")
                    continue
            
            # 生成响应
            print("\n助手思考中...", end="", flush=True)
            response = chat.generate_response(user_input, new_image)
            print("\r" + " " * 20 + "\r")  # 清除"思考中"消息
            print(f"\n助手: {response}")
            
        except KeyboardInterrupt:
            print("\n检测到中断，退出聊天...")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")

if __name__ == "__main__":
    interactive_chat()
