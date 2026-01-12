# export TEST_MODE=True

# # ============================================================================
# # BASELINE METHODS (원본)
# # ============================================================================

# # Baseline: greedy decoding
# # For "Controlled_Images_A", "Controlled_Images_B", "COCO_QA_one_obj", "COCO_QA_two_obj", use "four" option.
# # For "VG_QA_one_obj" and "VG_QA_two_obj", use "six" option.
# python3 main_aro.py \
#   --dataset=Controlled_Images_A \
#   --model-name='llava1.5' \
#   --download \
#   --method=base \
#   --option=four

# # Scaling_Vis (원본)
# # For Controlled_A: weight=0.8
# # For Controlled_B: weight=0.8
# # For COCO_QA_one_obj: weight=1.2
# # For COCO_QA_two_obj: weight=1.2
# # For VG_QA_one_obj: weight=2.0
# # For VG_QA_two_obj: weight=2.0
# # For VSR: weight=0.5
# python3 main_aro.py \
#   --dataset=Controlled_Images_A \
#   --model-name='llava1.5' \
#   --download \
#   --method=scaling_vis \
#   --weight=0.8 \
#   --option=four

# # Adapt_Vis (원본)
# # For Controlled_A: weight1=0.5, weight2=1.5, threshold=0.4
# # For Controlled_B: weight1=0.5, weight2=1.5, threshold=0.35
# # For COCO_QA_one_obj: weight1=0.5, weight2=1.2, threshold=0.3
# # For COCO_QA_two_obj: weight1=0.5, weight2=1.2, threshold=0.3
# # For VG_QA_one_obj: weight1=0.5, weight2=2.0, threshold=0.2
# # For VG_QA_two_obj: weight1=0.5, weight2=2.0, threshold=0.2
# # For VSR: weight1=0.5, weight2=1.2, threshold=0.64
# python3 main_aro.py \
#   --dataset=Controlled_Images_A \
#   --model-name='llava1.5' \
#   --download \
#   --method=adapt_vis \
#   --weight1=0.5 \
#   --weight2=1.5 \
#   --threshold=0.4 \
#   --option=four

# # ============================================================================
# # ADDITIONAL EXPERIMENTAL METHODS
# # ============================================================================

# # Adapt_Vis_JSD
# python3 main_aro.py \
#   --dataset=Controlled_Images_A \
#   --model-name='llava1.5' \
#   --download \
#   --method=adapt_vis_jsd \
#   --weight1=0.5 \
#   --weight2=1.5 \
#   --threshold=0.04 \
#   --option=four

# # Adapt_Vis_Obj
# python3 main_aro.py \
#   --dataset=Controlled_Images_A \
#   --model-name='llava1.5' \
#   --download \
#   --method=adapt_vis_obj \
#   --weight1=0.5 \
#   --weight2=1.5 \
#   --threshold=0.4 \
#   --option=four

# # Adapt_Vis_Entropy
# python3 main_aro.py \
#   --dataset=Controlled_Images_A \
#   --model-name='llava1.5' \
#   --download \
#   --method=adapt_vis_entropy \
#   --weight1=0.5 \
#   --weight2=1.5 \
#   --threshold=0.9 \
#   --option=four

# # Reasoning: Absolute 4 Directions
# python3 main_aro.py \
#   --dataset=Controlled_Images_A \
#   --model-name='llava1.5' \
#   --download \
#   --method=reasoning_absolute_4directions \
#   --option=four

# # Reasoning: Relative Location
# python3 main_aro.py \
#   --dataset=Controlled_Images_A \
#   --model-name='llava1.5' \
#   --download \
#   --method=reasoning_relative_location \
#   --option=four

# # Reasoning: Relative Relationship
# python3 main_aro.py \
#   --dataset=Controlled_Images_A \
#   --model-name='llava1.5' \
#   --download \
#   --method=reasoning_relative_relationship \
#   --option=four

# # Chain of Thought
# python3 main_aro.py \
#   --dataset=Controlled_Images_A \
#   --model-name='llava1.5' \
#   --download \
#   --method=chain_of_thought \
#   --option=four

# Oracle research
python3 main_aro.py \
    --dataset Controlled_Images_A \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_for_oracle_research \
    --weight1 0.5  \
    --weight2 1.5 \
    --threshold 0.4 \
    --option=four

python research.py \
    --results-path "output/results_Controlled_Images_A_adapt_vis_for_oracle_research_1.0_0.5_1.5_0.4_True.json"  \
    --save-prefix "output/analysis_Controlled_Images_A_True"