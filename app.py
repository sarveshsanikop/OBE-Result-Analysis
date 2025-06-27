import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def llm(prompt, max_tokens=300):
    return {
        "choices": [ {
            "text": "\n- Most unattempted questions are from higher Bloom Levels (e.g., L3) and target similar COs.\n- High non-attempt rates may indicate complexity or unfamiliar question formats.\n- Consider reinforcing key concepts and practicing with similar question styles.\n\nSummary: The non-attempted pattern suggests students may be struggling with application-level questions tied to specific outcomes. Adjusting teaching focus and question design can improve engagement."
        } ]
    }

# Global variables
temp_plot_path_1 = "plot_unattempted.png"
temp_plot_path_2 = "plot_attainment.png"
temp_plot_path_3 = "plot_strongest_weakest.png"
scores_df = None
metadata_df = None
percentage_attempted = None


def load_and_display_results(file_path):
    global scores_df, metadata_df, percentage_attempted
    try:
        df = pd.read_excel(file_path, header=None)

        metadata_df = df.iloc[:5, :].reset_index(drop=True)
        scores_df = df.iloc[5:, :].reset_index(drop=True)
        scores_df.columns = ["Student"] + metadata_df.iloc[0, 1:].astype(str).tolist()
        scores_df.iloc[:, 1:] = scores_df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

        max_marks = pd.to_numeric(metadata_df.iloc[4, 1:], errors="coerce").fillna(0).values
        scores_df = scores_df.iloc[1:].reset_index(drop=True)

        total_students = len(scores_df)
        attempted_counts = scores_df.iloc[:, 1:].notna().sum()
        total_scores = scores_df.iloc[:, 1:].sum()
        average_score = np.where(attempted_counts > 0, total_scores / attempted_counts, np.nan)
        percentage_attempted = (attempted_counts / total_students) * 100

        percent_attainment = np.where(
            (attempted_counts > 0) & (max_marks > 0),
            (average_score / max_marks) * 100,
            np.nan
        )
        total_possible_marks = np.where(max_marks > 0, total_students * max_marks, np.nan)
        actual_attainment = np.where(
            (total_possible_marks > 0),
            (total_scores / total_possible_marks) * 100,
            np.nan
        )

        average_score = [round(x, 2) if not np.isnan(x) else "-" for x in average_score]
        percentage_attempted = [round(x, 2) if not np.isnan(x) else "-" for x in percentage_attempted]
        percent_attainment = [round(x, 2) if not np.isnan(x) else "-" for x in percent_attainment]
        actual_attainment = [round(x, 2) if not np.isnan(x) else "-" for x in actual_attainment]

        new_rows = pd.DataFrame([
            ["Average Marks"] + average_score,
            ["Percentage"] + percentage_attempted,
            ["% Attainment"] + percent_attainment,
            ["Act Attainment"] + actual_attainment
        ])

        final_df = pd.concat([df, new_rows], ignore_index=True)
        return final_df

    except Exception as e:
        return f"An error occurred: {e}"

def display_results():
    global scores_df, metadata_df, percentage_attempted
    numeric_percentage_attempted = pd.to_numeric(pd.Series(percentage_attempted), errors='coerce')
    non_attempt_percentage = 100 - numeric_percentage_attempted

    question_meta = pd.DataFrame({
        "Question": scores_df.columns[1:],
        "PI": metadata_df.iloc[1, 1:].values,
        "Bloom_Level": metadata_df.iloc[2, 1:].values,
        "CO": metadata_df.iloc[3, 1:].values,
        "Max_Marks": metadata_df.iloc[4, 1:].values,
        "% Attempted": numeric_percentage_attempted,
        "% Not Attempted": non_attempt_percentage
    })

    threshold = 50
    unattempted_questions = question_meta[question_meta["% Not Attempted"] >= threshold]
    unattempted_questions = unattempted_questions.sort_values(by="% Not Attempted", ascending=False).reset_index(drop=True)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(unattempted_questions['Question'], unattempted_questions['% Not Attempted'], color='salmon')
    plt.xlabel('Question')
    plt.ylabel('% Not Attempted')
    plt.title('Questions with High Non-Attempt Rate')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(temp_plot_path_1)
    plt.close()

    # Insights
    unattempted_text = unattempted_questions.head(5).to_string(index=False)
    prompt = f"""
You are an academic analyst preparing observations for a university internal exam review.

Dataset – Questions with High Non-Attempt Rate:
{unattempted_text}

Column Descriptions:
- Question: Question number
- PI: Performance Indicator
- Bloom_Level: Bloom's taxonomy level (e.g., L2 = Understand, L3 = Apply)
- CO: Course Outcome number
- Max_Marks: Maximum marks allocated to the question
- % Attempted: Percentage of students who attempted the question
- % Not Attempted: Percentage of students who left it unattempted

*Your task:*
- Write 1–2 line bullet points:
  1. Identify common traits among unattempted questions (e.g., Bloom_Level, CO, PI).
  2. Suggest reasons for high non-attempt rates.
  3. Recommend ways to improve question design or student preparation.
- Conclude with a short summary (2–3 lines).

Avoid assumptions not supported by the table.

Now provide your insights:
"""
    output = llm(prompt=prompt, max_tokens=300)
    insights_text = output['choices'][0]['text'].strip()
    return unattempted_questions, temp_plot_path_1, insights_text

custom_css = """
.card {
    background-color: #f0f0f0;
    border: 2px solid darkred;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    transition: transform 0.3s ease;
}
.card:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 10px rgba(255, 0, 0, 0.5);
}
.red-btn {
    background-color: #8B0000 !important;
    color: white !important;
}
img.logo {
    width: 60px !important;
    height: auto !important;
    object-fit: contain;
    margin-right: 10px;
}
h1, h2 {
    font-size: 36px;
}

/* NEW: Blue GEN AI text styling */
.powered-by-genai {
    color: #0056D2 !important;
    text-align: right;
    font-size: 16px;
    padding-top: 30px;
    font-weight: bold;
}
"""


def display_top_attainment():
    global scores_df, metadata_df

    max_marks = pd.to_numeric(metadata_df.iloc[4, 1:], errors="coerce").fillna(0).values
    attempted_counts = scores_df.iloc[:, 1:].notna().sum()
    total_scores = scores_df.iloc[:, 1:].sum()
    average_score = np.where(attempted_counts > 0, total_scores / attempted_counts, np.nan)

    percent_attainment = np.where(
        (attempted_counts > 0) & (max_marks > 0),
        (average_score / max_marks) * 100,
        np.nan
    )

    attainment_df = pd.DataFrame({
        "Question": scores_df.columns[1:],
        "PI": metadata_df.iloc[1, 1:].values,
        "Bloom_Level": metadata_df.iloc[2, 1:].values,
        "CO": metadata_df.iloc[3, 1:].values,
        "% Attainment": percent_attainment
    })

    top5 = attainment_df.sort_values(by="% Attainment", ascending=False).head(5)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Question", y="% Attainment", data=top5, palette="viridis")
    plt.title("Top 5 Questions by % Attainment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = "top5_attainment_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # Simple insights (placeholder or use LLM like others)
    insights = "\n".join([
        "- These are the top 5 questions based on percentage attainment.",
        "- They reflect areas where students performed best.",
        "- Consider using similar question styles in future assessments."
    ])

    return top5, plot_path, insights

# Ensure display_strongest_weakest returns insights_text
def display_strongest_weakest():
    global scores_df, metadata_df

    max_marks = pd.to_numeric(metadata_df.iloc[4, 1:], errors="coerce").fillna(0).values
    attempted_counts = scores_df.iloc[:, 1:].notna().sum()
    total_scores = scores_df.iloc[:, 1:].sum()
    average_score = np.where(attempted_counts > 0, total_scores / attempted_counts, np.nan)

    percent_attainment = np.where(
        (attempted_counts > 0) & (max_marks > 0),
        (average_score / max_marks) * 100,
        np.nan
    )

    all_attainment_df = pd.DataFrame({
        "Question": scores_df.columns[1:],
        "PI": metadata_df.iloc[1, 1:].values,
        "Bloom_Level": metadata_df.iloc[2, 1:].values,
        "CO": metadata_df.iloc[3, 1:].values,
        "Max_Marks": max_marks,
        "% Attainment": percent_attainment
    }).reset_index(drop=True)

    weakest = all_attainment_df.sort_values(by="% Attainment").head(1)
    strongest = all_attainment_df.sort_values(by="% Attainment", ascending=False).head(1)

    # Prepare data for insights
    weak_text = weakest.to_string(index=False)
    strong_text = strongest.to_string(index=False)

    # Prompt for LLM (Insights Generation)
    prompt = f"""
    You are an academic analyst preparing a short performance summary for an internal university assessment report.

    Dataset – Weakest Question:
    {weak_text}

    Dataset – Strongest Question:
    {strong_text}

    Column Descriptions:
    - Question: Question number
    - PI: Performance Indicator
    - Bloom_Level: Bloom's taxonomy level (e.g., L2 = Understand, L3 = Apply)
    - CO: Course Outcome number
    - Max_Marks: Maximum marks assigned to the question
    - % Attainment: Percentage of total marks achieved by students

    **Your task:**
    - Write clear, 1–2 line bullet points:
      1. Highlight key differences between the strongest and weakest question.
      2. Analyze based on Bloom_Level and CO.
      3. Suggest short, actionable improvement strategies.
    - End with a brief 2–3 line conclusion.

    Do not make up extra details. Only use the table data.

    Now provide your insights:
    """
    output = llm(prompt=prompt, max_tokens=300)
    insights_text = output['choices'][0]['text'].strip()

    # Plotting the Strongest vs Weakest Questions
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Question", y="% Attainment", data=all_attainment_df, palette="coolwarm")
    plt.axvline(x=weakest.index[0], color='red', linestyle='--', label='Weakest Question')
    plt.axvline(x=strongest.index[0], color='green', linestyle='--', label='Strongest Question')
    plt.title("Strongest vs Weakest Question Performance")
    plt.xlabel("Question")
    plt.ylabel("% Attainment")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plot_path = "strongest_weakest_analysis.png"
    plt.savefig(plot_path)
    plt.close()

    return strongest, weakest, plot_path, insights_text  # Return the insights as well
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def attempt_vs_score_analysis():
    max_marks = pd.to_numeric(metadata_df.iloc[4, 1:], errors="coerce").fillna(0).values
    attempted_counts = scores_df.iloc[:, 1:].notna().sum()
    total_students = len(scores_df)
    total_scores = scores_df.iloc[:, 1:].sum()

    average_scores = np.where(attempted_counts > 0, total_scores / attempted_counts, np.nan)
    percent_attempted = (attempted_counts / total_students) * 100

    analysis_df = pd.DataFrame({
        "Question": scores_df.columns[1:],
        "PI": metadata_df.iloc[1, 1:].values,
        "Bloom_Level": metadata_df.iloc[2, 1:].values,
        "CO": metadata_df.iloc[3, 1:].values,
        "Max_Marks": max_marks,
        "Avg_Score": np.round(np.array(average_scores, dtype=float), 2),
        "% Attempted": np.round(np.array(percent_attempted, dtype=float), 2)
    })

    avg_score_threshold = np.nanmean(average_scores)
    attempt_rate_threshold = np.nanmean(percent_attempted)

    low_avg_high_attempt = analysis_df[
        (analysis_df["Avg_Score"] < avg_score_threshold) &
        (analysis_df["% Attempted"] >= attempt_rate_threshold)
    ].sort_values(by="Avg_Score").head(5)

    high_avg_low_attempt = analysis_df[
        (analysis_df["Avg_Score"] >= avg_score_threshold) &
        (analysis_df["% Attempted"] < attempt_rate_threshold)
    ].sort_values(by="Avg_Score", ascending=False).head(5)

    # --- Scatter Plot ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="% Attempted", y="Avg_Score", data=analysis_df,
        hue="CO", style="Bloom_Level", s=100, palette="tab10"
    )
    plt.axhline(avg_score_threshold, color='r', linestyle='--', label='Avg Score Threshold')
    plt.axvline(attempt_rate_threshold, color='b', linestyle='--', label='Attempt Rate Threshold')
    plt.title("Avg Score vs % Attempted (Colored by CO, Shaped by Bloom Level)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    scatter_path = "score_vs_attempt.png"
    plt.savefig(scatter_path)
    plt.close()

    # --- Bar Plot: Avg Score per Question ---
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Question", y="Avg_Score", data=analysis_df, palette="Blues_d")
    plt.title("Average Score per Question")
    plt.xticks(rotation=45)
    plt.ylabel("Average Score")
    plt.tight_layout()
    avg_score_plot_path = "avg_score_per_question.png"
    plt.savefig(avg_score_plot_path)
    plt.close()

    # --- Bar Plot: % Attempted per Question ---
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Question", y="% Attempted", data=analysis_df, palette="Greens_d")
    plt.title("Percentage of Students Attempted per Question")
    plt.xticks(rotation=45)
    plt.ylabel("% Attempted")
    plt.tight_layout()
    attempt_plot_path = "percent_attempted_per_question.png"
    plt.savefig(attempt_plot_path)
    plt.close()

    # --- Insights ---
    insights = (
        f"🔍 **Low Avg Score & High Attempt Rate**: These questions were commonly attempted but had poor performance.\n"
        f"🎯 **High Avg Score & Low Attempt Rate**: These questions were less attempted but had high performance.\n\n"
        f"- Consider revisiting the low-performing, high-attempt questions for teaching gaps.\n"
        f"- Consider why some high-performing questions are avoided — difficulty? perception?\n"
    )

    return (
        low_avg_high_attempt,
        high_avg_low_attempt,
        scatter_path,
        avg_score_plot_path,
        attempt_plot_path,
        insights
    )
  
with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(
                value="/content/drive/My Drive/KLE_Technological_University_Enhanced[1].png",
                show_label=False,
                elem_classes="logo"
            )
        with gr.Column(scale=8):
            gr.Markdown(
                "## **KLE Technological University**  \n### *ESA Results Analysis Tool*"
            )
        with gr.Column(scale=3):
            gr.Markdown("**Powered by GEN AI**", elem_classes="powered-by-genai")

    gr.Markdown("---")






    # Row for sections (side by side layout)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Upload ESA Excel File")
            file_input = gr.File(label="Upload File", type="filepath")
            display_btn = gr.Button("Load & Display Results", elem_classes="red-btn")
            result_output = gr.DataFrame(label="Raw + Computed Result Sheet")
            display_btn.click(load_and_display_results, inputs=file_input, outputs=result_output)

        # Non-Attempted Questions Analysis - Initially hidden
        with gr.Column(scale=3, elem_classes="card", visible=False) as non_attempted_section:
            gr.Markdown("### ❌ Analysis of Non-Attempted Questions")
            table_output = gr.DataFrame(label="Unattempted Questions")
            plot_output = gr.Image(type="filepath", label="Non-Attempt Rate Plot")
            insights_output = gr.Textbox(label="Generated Insights", lines=10, interactive=False)
            gr.Button("Generate Non-Attempted Analysis", elem_classes="red-btn").click(
                display_results, outputs=[table_output, plot_output, insights_output]
            )

        # Top 5 Attainment Analysis - Initially hidden
        with gr.Column(scale=3, elem_classes="card", visible=False) as top_attainment_section:
            gr.Markdown("### ✅ Top 5 Questions by % Attainment")
            attainment_table = gr.DataFrame(label="Top 5 Attained Questions")
            attainment_plot = gr.Image(type="filepath", label="Top Attainment Bar Chart")
            attainment_insights = gr.Textbox(label="Generated Insights", lines=10, interactive=False)
            gr.Button("Show Top 5 Attainment", elem_classes="red-btn").click(
                display_top_attainment, outputs=[attainment_table, attainment_plot, attainment_insights]
            )

        # Strongest vs Weakest Questions Analysis - Initially hidden
        with gr.Column(scale=3, elem_classes="card", visible=False) as strongest_weakest_section:
            gr.Markdown("### 🎯 Strongest vs Weakest Question Analysis")
            strongest_table = gr.DataFrame(label="Strongest Question")
            weakest_table = gr.DataFrame(label="Weakest Question")
            plot_output_3 = gr.Image(type="filepath", label="Strongest vs Weakest Analysis Plot")
            insights_output_3 = gr.Textbox(label="Generated Insights", lines=10, interactive=False)
            gr.Button("Generate Strongest & Weakest Analysis", elem_classes="red-btn").click(
                display_strongest_weakest, outputs=[strongest_table, weakest_table, plot_output_3, insights_output_3]
            )

        # Attempt vs Score Analysis - Initially hidden
        with gr.Column(scale=3, elem_classes="card", visible=False) as attempt_vs_score_section:
            gr.Markdown("### 🔍 Low Avg Score & High Attempt vs High Avg Score & Low Attempt Analysis")
            with gr.Row():
                low_table = gr.DataFrame(label="🔻 Low Avg Score & High Attempt")
                high_table = gr.DataFrame(label="🚀 High Avg Score & Low Attempt")
            with gr.Row():
                scatter_plot = gr.Image(label="Score vs Attempt Scatter", type="filepath")
                avg_plot = gr.Image(label="Average Score per Question", type="filepath")
                attempt_plot = gr.Image(label="% Attempted per Question", type="filepath")
            analysis_insights = gr.Textbox(label="Insights", lines=6, interactive=False)

            gr.Button("Run Attempt vs Score Analysis", elem_classes="red-btn").click(
                attempt_vs_score_analysis,
                outputs=[low_table, high_table, scatter_plot, avg_plot, attempt_plot, analysis_insights]
            )

    # Buttons Row – Aligned side by side
    with gr.Row():
        show_non_attempted_button = gr.Button("📌 View Unattempted Questions", elem_classes="red-btn")
        show_top_attainment_button = gr.Button("🏆 View Top 5 Attained Questions", elem_classes="red-btn")
        show_strongest_weakest_button = gr.Button("📊 View Strongest & Weakest Questions", elem_classes="red-btn")
        show_attempt_vs_score_button = gr.Button("📈 View Attempt vs Score Insights", elem_classes="red-btn")

    # Button click events
    show_non_attempted_button.click(
        lambda: [gr.update(visible=True), gr.update(visible=False)],
        outputs=[non_attempted_section, show_non_attempted_button]
    )

    show_top_attainment_button.click(
        lambda: [gr.update(visible=True), gr.update(visible=False)],
        outputs=[top_attainment_section, show_top_attainment_button]
    )

    show_strongest_weakest_button.click(
        lambda: [gr.update(visible=True), gr.update(visible=False)],
        outputs=[strongest_weakest_section, show_strongest_weakest_button]
    )

    show_attempt_vs_score_button.click(
        lambda: [gr.update(visible=True), gr.update(visible=False)],
        outputs=[attempt_vs_score_section, show_attempt_vs_score_button]
    )

gr.Markdown("---")
gr.Markdown("© 2025 KLE Technological University | Department of CSE")

demo.launch()
