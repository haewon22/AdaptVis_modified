import json
import sys
from pathlib import Path

def extract_direction_absolute(text):
    """Extract direction keyword for absolute reasoning"""
    text = text.lower()
    if 'left' in text:
        return 'left'
    elif 'right' in text:
        return 'right'
    elif 'top' in text or 'above' in text or 'upper' in text:
        return 'top'
    elif 'bottom' in text or 'below' in text or 'under' in text or 'lower' in text:
        return 'bottom'
    return None

def extract_direction_relative(text):
    """Extract direction keyword for relative reasoning"""
    text = text.lower()
    if 'under' in text or 'below' in text or 'beneath' in text:
        return 'under'
    elif 'on' in text and 'front' not in text:
        return 'on'
    elif 'left' in text:
        return 'left'
    elif 'right' in text:
        return 'right'
    return None

def check_reasoning_results(result_file):
    """Check reasoning method results"""
    print(f"\n{'='*60}")
    print(f"Checking: {result_file}")
    print(f"{'='*60}\n")
    
    if not Path(result_file).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {result_file}")
        return
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    print(f"총 샘플 수: {len(data)}")
    
    # Check for reasoning-specific fields
    has_step1 = 'Step1_1_to_2' in data[0] if data else False
    has_consistent = 'Is_consistent' in data[0] if data else False
    
    if has_step1 and has_consistent:
        # Determine reasoning type
        reasoning_type = None
        if data:
            first_step1 = data[0].get('Step1_1_to_2', '')
            if any(word in first_step1.lower() for word in ['top', 'bottom']):
                reasoning_type = 'absolute'
            else:
                reasoning_type = 'relative'
        
        # Absolute or Relative Location reasoning
        consistent_count = sum(1 for x in data if x.get('Is_consistent', False))
        print(f"\n✓ Reasoning 메서드 감지됨 (Type: {reasoning_type})")
        print(f"  - Is_consistent=True: {consistent_count}/{len(data)} ({consistent_count/len(data)*100:.1f}%)")
        print(f"  - Is_consistent=False: {len(data)-consistent_count}/{len(data)} ({(len(data)-consistent_count)/len(data)*100:.1f}%)")
        
        if consistent_count == 0:
            print(f"\n⚠️  경고: 모든 샘플에서 consistent=False")
            print(f"   → 키워드 추출 실패 가능성")
        else:
            print(f"\n✓ Reasoning이 작동 중! (consistent > 0)")
        
        # Check keyword extraction
        extract_fn = extract_direction_absolute if reasoning_type == 'absolute' else extract_direction_relative
        
        extraction_success = 0
        for item in data:
            step1_1 = item.get('Step1_1_to_2', '')
            step1_2 = item.get('Step1_2_to_1', '')
            
            dir1 = extract_fn(step1_1)
            dir2 = extract_fn(step1_2)
            
            if dir1 is not None and dir2 is not None:
                extraction_success += 1
        
        print(f"\n키워드 추출 성공률: {extraction_success}/{len(data)} ({extraction_success/len(data)*100:.1f}%)")
        
        # Show examples
        print(f"\n샘플 예시 (처음 3개):")
        for i, item in enumerate(data[:3]):
            step1_1 = item.get('Step1_1_to_2', 'N/A')
            step1_2 = item.get('Step1_2_to_1', 'N/A')
            
            dir1 = extract_fn(step1_1)
            dir2 = extract_fn(step1_2)
            
            print(f"\n[샘플 {i+1}]")
            print(f"  Prompt: {item.get('Prompt', '')[:80]}...")
            print(f"  Step1_1_to_2: {step1_1}")
            print(f"    → Extracted: {dir1}")
            print(f"  Step1_2_to_1: {step1_2}")
            print(f"    → Extracted: {dir2}")
            print(f"  Is_consistent: {item.get('Is_consistent', 'N/A')}")
            print(f"  Generation: {item.get('Generation', 'N/A')}")
            print(f"  Golden: {item.get('Golden', 'N/A')}")
    
    elif 'Step1' in data[0] if data else False:
        # Relative Relationship reasoning
        print(f"\n✓ Reasoning Relative Relationship 감지됨")
        print(f"\n샘플 예시:")
        for i, item in enumerate(data[:3]):
            print(f"\n[샘플 {i+1}]")
            print(f"  Step1: {item.get('Step1', 'N/A')}")
            print(f"  Generation: {item.get('Generation', 'N/A')}")
            print(f"  Golden: {item.get('Golden', 'N/A')}")
    
    else:
        print(f"\n❌ Reasoning 메서드가 아니거나 구조가 다릅니다")
        print(f"   사용 가능한 필드: {list(data[0].keys()) if data else []}")
    
    # Check accuracy
    correct = 0
    for item in data:
        gen = item.get('Generation', '').lower()
        golden = item.get('Golden', '').lower()
        
        if golden in gen and not (golden == 'on' and 'front' in gen):
            correct += 1
    
    accuracy = correct / len(data) if data else 0
    print(f"\n정확도: {correct}/{len(data)} ({accuracy*100:.2f}%)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 check_reasoning.py <result_file.json>")
        print("\nExample:")
        print("  python3 check_reasoning.py output/results1.5_Controlled_Images_A_reasoning_absolute_4directions_*.json")
        sys.exit(1)
    
    for result_file in sys.argv[1:]:
        check_reasoning_results(result_file)
