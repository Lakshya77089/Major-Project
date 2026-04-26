"""
Generate a Hinglish explainer PDF for the zkFedMoE project.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
)


OUT = Path(__file__).parent / "zkFedMoE_Hinglish_Explainer.pdf"


def build_styles():
    s = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleX", parent=s["Title"], fontSize=22, leading=26,
        textColor=colors.HexColor("#1a3d7c"), spaceAfter=6,
    )
    subtitle = ParagraphStyle(
        "Sub", parent=s["Normal"], fontSize=11, leading=14,
        textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=18,
    )
    h1 = ParagraphStyle(
        "H1", parent=s["Heading1"], fontSize=15, leading=19,
        textColor=colors.HexColor("#1a3d7c"), spaceBefore=10, spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2", parent=s["Heading2"], fontSize=12, leading=15,
        textColor=colors.HexColor("#2e5fa0"), spaceBefore=8, spaceAfter=4,
    )
    body = ParagraphStyle(
        "Body", parent=s["Normal"], fontSize=10, leading=14,
        spaceAfter=6, alignment=4,  # justify
    )
    bullet = ParagraphStyle(
        "Bul", parent=s["Normal"], fontSize=10, leading=14,
        leftIndent=14, bulletIndent=2, spaceAfter=3,
    )
    code = ParagraphStyle(
        "Code", parent=s["Code"], fontSize=9, leading=12,
        textColor=colors.HexColor("#222222"),
        backColor=colors.HexColor("#f4f4f4"),
        leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=8,
    )
    note = ParagraphStyle(
        "Note", parent=s["Normal"], fontSize=9, leading=12,
        textColor=colors.HexColor("#333333"),
        backColor=colors.HexColor("#fff8d9"),
        leftIndent=8, rightIndent=8, borderPadding=6, spaceAfter=8,
    )
    return {
        "title": title, "subtitle": subtitle,
        "h1": h1, "h2": h2, "body": body, "bullet": bullet,
        "code": code, "note": note,
    }


def bullet_list(items, st):
    return [Paragraph(f"&bull; {x}", st["bullet"]) for x in items]


def main():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.6*cm, bottomMargin=1.6*cm,
    )
    st = build_styles()
    flow = []
    P = lambda text, style="body": flow.append(Paragraph(text, st[style]))
    SP = lambda h=8: flow.append(Spacer(1, h))

    # ---------- COVER ----------
    P("zkFedMoE — Hinglish Explainer", "title")
    P(
        "Zero-Knowledge Federated Mixture-of-Experts<br/>"
        "Major Project | Group #34 | IIIT Kota | April 2026",
        "subtitle",
    )

    P("1. Project ek line me kya hai?", "h1")
    P(
        "<b>zkFedMoE</b> ek <b>Federated Learning</b> system hai jo <b>privacy</b> "
        "(DP-SGD), <b>verifiability</b> (SEPG proofs), aur <b>communication "
        "efficiency</b> (Top-K MoE) — teeno ko ek saath solve karta hai. "
        "Matlab: kayi clients (jaise hospitals/phones) apna data share kiye bina "
        "ek shared AI model train karte hain, aur server proof check karke maan "
        "leta hai ki sab kuch sahi se hua."
    )

    P("Real-world analogy", "h2")
    P(
        "Socho 5 hospitals hain. Sab apne marizon ka data share nahi kar sakte "
        "(privacy law). Lekin sab ek hi diagnostic AI ko milke better banana "
        "chahte hain. zkFedMoE bolta hai — koi data bhej hi mat, sirf <i>model "
        "updates</i> bhejo, aur unme bhi noise daal do (DP), aur ek SHA-256 proof "
        "bhejo ki tumne sahi rules follow kiye. Server proof verify karke updates "
        "ko average kar deta hai."
    )

    P("2. App ko start kaise karein?", "h1")
    P("Step 1 — Dependencies install (one-time)", "h2")
    P(
        '<font face="Courier">cd "d:/projects/Major Project/Code"<br/>'
        "pip install torch pandas numpy streamlit plotly matplotlib graphviz reportlab</font>",
        "code",
    )
    P("Step 2 — Tin tarike se chalao (apni zarurat ke hisaab se)", "h2")

    table_data = [
        ["Tarika", "Command", "Kab use karein"],
        [
            "Dashboard\n(recommended)",
            "python -m streamlit run dashboard.py",
            "Viva, demo, ya interactive exploration",
        ],
        [
            "Full experiments",
            "python -m experiments.run_all_experiments",
            "Saare 4 experiments ki plots banane ke liye (~10-15 min CPU)",
        ],
        [
            "Quick demo",
            "python run_demo.py",
            "Phase 1+2 ka end-to-end ek baar mein dekhne ke liye (~3 min)",
        ],
    ]
    t = Table(table_data, colWidths=[3.2*cm, 6.5*cm, 6.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d7c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.HexColor("#f4f7fb"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    flow.append(t)
    SP(10)
    P(
        "Dashboard browser mein <font face=\"Courier\">http://localhost:8501</font> "
        "par khulta hai. 10 pages hain — Home se start karke About tak.",
        "note",
    )

    flow.append(PageBreak())

    # ---------- BIG PICTURE ----------
    P("3. Pura system kaise kaam karta hai? (One Round)", "h1")
    P(
        "Federated Learning mein ek &lsquo;round&rsquo; matlab ek poora cycle. "
        "Hamare project mein 5 rounds default hain. Har round mein 8 steps:"
    )
    steps = [
        "<b>Server</b> apna global model (theta_t) sab clients ko bhejta hai.",
        "<b>Client</b> apne private data par local training karta hai (data kabhi nahi nikalta).",
        "Top-K experts select karta hai — sab experts ki jagah sirf K most-used.",
        "<b>DP-SGD</b> apply: gradient clip karo (max norm C), Gaussian noise daalo (sigma).",
        "<b>SEPG proof</b> banao: TopK list + clip_norm + sigma + SHA-256 hash.",
        "Sparse update + proof server ko bhejo (sirf bytes, data nahi).",
        "<b>Server</b> 4-check verify karta hai — fail hua to client reject.",
        "Sab pass hue updates ko aggregate karo (FedAvg / Median / TrimMean) &rArr; theta_(t+1).",
    ]
    flow.extend(bullet_list(steps, st))
    SP(8)
    P(
        "<b>Key insight:</b> Raw data network par kabhi travel nahi karta. "
        "Sirf model gradients aur ek chhota proof. Yahi privacy-preserving FL ka core hai.",
        "note",
    )

    # ---------- FUNCTIONALITIES ----------
    P("4. Saari functionalities — kya, kyu, kahan", "h1")

    feats = [
        ("MoE + LoRA model",
         "Tiny model hai (~600K params). MoE ke 8 experts hain, har input ke liye "
         "router decide karta hai konse 2 experts use karne hain (Top-2). LoRA "
         "classifier head full Linear ki jagah low-rank A,B matrices use karta "
         "hai &mdash; trainable params 100x kam.",
         "src/models/moe_model.py"),
        ("AG News Dataset",
         "120K news headlines, 4 classes (World, Sports, Business, Tech). "
         "5 simulated clients me split. Optionally Dirichlet(alpha) split for "
         "non-IID setting (real FL jaise).",
         "src/data/text_datasets.py"),
        ("FedAvg client local training",
         "Har client apne shard par 2 epochs train karta hai. Adam optimizer, "
         "cross-entropy loss. FedProx mu>0 set kar do to non-IID drift kam.",
         "src/fl/client.py"),
        ("Top-K Sparse Communication",
         "Client sirf Top-K experts ke weights bhejta hai, baaki ke nahi. "
         "K=2 of E=8 &rArr; ~75% bandwidth saving. Saving = 1 - K/E.",
         "src/fl/client.py"),
        ("Differential Privacy (DP-SGD)",
         "clip_update() &rarr; gradient L2 norm ko C par cap karta hai. "
         "add_noise() &rarr; Gaussian N(0, sigma^2) daalta hai. "
         "Renyi accountant (alpha,sigma,q,T) se tight (epsilon, delta) bound deta hai.",
         "src/fl/dp.py"),
        ("SEPG Proof + 4-check Verify",
         "Client banata hai: TopK list, ||grad||, sigma, SHA-256(grad). "
         "Server check karta hai: (1) |TopK|=K, (2) clip_norm <= max, "
         "(3) sigma >= sigma_min, (4) hash match. Fail = reject client.",
         "src/fl/sepg.py"),
        ("Secure Aggregation (pairwise masks)",
         "Bonawitz et al. 2017 ka simplified version. Har client apni update "
         "mein deterministic masks add karta hai jo sum karne par cancel ho "
         "jaate hain. Server ko sum dikhta hai, individual update nahi.",
         "src/fl/server.py (aggregate_secure)"),
        ("Adversary Simulations",
         "3 attacks: <b>Poisoning</b> (labels flip), <b>Free-rider</b> (jhoothi "
         "training, sirf noise), <b>Sybil</b> (1 attacker, N fake identities). "
         "Real FL deployment mein ye sab khatre real hain.",
         "src/fl/adversaries.py"),
        ("Robust Aggregation",
         "FedAvg easy hai par poisoning ke against weak. "
         "<b>Coordinate-wise Median</b> = har parameter ka median, byzantine-robust. "
         "<b>Trimmed Mean</b> = top/bottom k% drop kar ke average. "
         "40% malicious par Median best (only -7pp drop).",
         "src/fl/server.py"),
        ("Verifiable Oracle Committee",
         "M oracle nodes har client ki update ko independent score karte hain. "
         "Final decision = median score (M/2 oracles galat ho to bhi safe). "
         "Blockchain-free version of threshold signatures.",
         "src/fl/sepg.py (committee_decision)"),
        ("4 Automated Experiments",
         "Exp1: Privacy-utility tradeoff (sigma vs accuracy). "
         "Exp2: Communication savings vs K. "
         "Exp3: SEPG verification overhead (~6ms). "
         "Exp4: Robustness under 0/20/40% poisoning.",
         "experiments/run_all_experiments.py"),
        ("10-page Streamlit Dashboard",
         "Home, Predict (live text classify), Train (configurable FL), Custom CSV "
         "(upload your data), Privacy &amp; DP (live epsilon graph), Robustness, "
         "Experiments, Compare, Architecture, About.",
         "dashboard.py"),
    ]

    feat_rows = [["#", "Feature", "Kya karta hai (Hinglish)", "File"]]
    for i, (n, d, f) in enumerate(feats, 1):
        feat_rows.append([str(i), Paragraph(f"<b>{n}</b>", st["body"]),
                          Paragraph(d, st["body"]),
                          Paragraph(f'<font face="Courier" size="8">{f}</font>',
                                    st["body"])])

    ft = Table(feat_rows, colWidths=[0.7*cm, 3.8*cm, 8.0*cm, 4.0*cm])
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d7c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.HexColor("#f4f7fb"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(ft)

    flow.append(PageBreak())

    # ---------- DASHBOARD PAGES ----------
    P("5. Dashboard ke 10 pages — har page kya karta hai", "h1")
    pages = [
        ("Home", "Pipeline ka graphviz diagram, concept cards, demo walkthrough. "
                 "Yahi se start karna hai."),
        ("Predict", "Apna text type karo (e.g. \"Apple unveils new iPhone\") &rarr; "
                    "model live classify karega + dikhayega konse 2 experts route hue."),
        ("Train", "Configurable FL training. Rounds, clients, K, DP on/off — "
                  "sab slider/toggles. Live charts."),
        ("Custom CSV", "Apni CSV upload karo (label, text columns). FL train hoga, "
                       "confusion matrix dikhega."),
        ("Privacy &amp; DP", "DP-SGD ke saath training. Live (epsilon, delta) curve. "
                             "SEPG proof JSON dikhayega."),
        ("Robustness", "Poisoning/Freerider/Sybil attack simulate karo. "
                        "FedAvg vs Median vs TrimMean compare hoga."),
        ("Experiments", "Saare 4 pre-computed experiments ke interactive Plotly charts."),
        ("Compare", "Real-time communication savings calculator (slider K)."),
        ("Architecture", "Model data-flow, parameter pie chart, 5 code-snippet tabs."),
        ("About", "Team, advisor, institution, status table."),
    ]
    for n, d in pages:
        P(f"&bull; <b>{n}</b> &mdash; {d}", "bullet")

    SP(8)

    # ---------- MATH ----------
    P("6. Math sirf zarurat ke (cheat-sheet for viva)", "h1")

    math_pairs = [
        ("FedAvg",
         "theta_(t+1) = sum_i (n_i / n) * theta_i_(t)<br/>"
         "Simply: clients ke models ka weighted average, weight = data size."),
        ("Top-K MoE",
         "y = sum over Top-K experts of g_k(x) * E_k(x)<br/>"
         "Router 8 experts mein se 2 chunta hai, baaki compute mat karo."),
        ("LoRA",
         "W' = W + B*A, where B in R^(d x r), A in R^(r x d), r &lt;&lt; d<br/>"
         "Full W ki jagah chote A,B train karo. Param count 100x kam."),
        ("DP-SGD: clip + noise",
         "gbar = g / max(1, ||g||/C)   &mdash; gradient ko cap karo<br/>"
         "g_tilde = (1/B) * (sum gbar + N(0, sigma^2 * C^2 * I))   &mdash; noise daalo"),
        ("(epsilon, delta)-DP",
         "Pr[M(D) in S] &lt;= e^epsilon * Pr[M(D') in S] + delta<br/>"
         "D, D' me 1 record ka difference. Smaller epsilon = stronger privacy."),
        ("Renyi Accountant",
         "epsilon(alpha) = (alpha / (2*sigma^2)) * T * q^2<br/>"
         "T rounds aur sampling rate q ke saath cumulative privacy loss."),
        ("Comm Saving",
         "Saving = 1 - K/E. K=2, E=8 &rArr; 75%."),
        ("Coordinate-wise Median",
         "theta_(t+1)[j] = median(theta_1[j], ..., theta_N[j])<br/>"
         "Per-parameter median &rArr; outlier (poisoned) clients ka effect 0."),
    ]
    for k, v in math_pairs:
        P(f"<b>{k}</b><br/><font face=\"Courier\" size=\"9\">{v}</font>", "code")

    # ---------- RESULTS ----------
    P("7. Results — kya prove hua", "h1")

    res = [
        ["Experiment", "Result", "Interpretation"],
        ["1. Privacy-Utility",
         "no DP &rarr; 58%, sigma=0.1 &rArr; ~25%",
         "DP correctly accounted; small-scale (5 rounds) accuracy cliff expected"],
        ["2. Comm Savings (K)",
         "K=4 best (59% acc, 23% saving)",
         "Sweet spot K=3-4. K=8 over-fits (no implicit reg)"],
        ["3. SEPG Overhead",
         "~6 ms/client, K-independent",
         "Negligible vs training time"],
        ["4. Robustness (40% mal.)",
         "FedAvg -16pp, Median -7pp, TrimMean -9pp",
         "Median is most byzantine-robust"],
    ]
    rt = Table(res, colWidths=[3.5*cm, 5.0*cm, 7.5*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d7c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.HexColor("#f4f7fb"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(rt)
    SP(10)

    # ---------- WHY EACH PIECE ----------
    P("8. Har piece kyu zaruri hai? (Viva ka Q&amp;A)", "h1")

    qa = [
        ("Q. FL hi kyu, central training kyu nahi?",
         "Privacy laws (GDPR, HIPAA) raw data centralize karne nahi dete. "
         "FL me data device se nikalti hi nahi."),
        ("Q. MoE kyu use kiya, simple FFN kyu nahi?",
         "MoE ke saath har client ko sirf K experts share karne padte hain, "
         "saare nahi &rArr; communication O(N) se O(K) ho jaata hai. Ye paper ka USP hai."),
        ("Q. LoRA ka kya role hai?",
         "Trainable parameters drastically reduce karta hai. Edge devices "
         "(phones) par memory aur compute kam hai &mdash; LoRA fits."),
        ("Q. DP-SGD me clip pehle kyu, noise pehle kyu nahi?",
         "Clip pehle kyunki gradient sensitivity bound karna hai. Sensitivity "
         "= C ho gayi, ab noise sigma*C add karenge &rArr; (epsilon, delta) guarantee."),
        ("Q. SEPG zero-knowledge kaise hai?",
         "Hum cryptographic ZK-SNARK use nahi karte (heavy hai). Hum SHA-256 "
         "commitment + structural checks karte hain &mdash; lightweight ZK-style "
         "verification. Server gradient nahi dekh sakta, sirf properties check karta hai."),
        ("Q. Median FedAvg se kyu better under attack?",
         "FedAvg outliers se kheech jaata hai (mean is sensitive). Median "
         "byzantine-robust hai &mdash; up to N/2 malicious clients tak unaffected."),
        ("Q. Hum 100 patients use karte hain kya?",
         "Nahi. Hum AG News (120K news headlines, 4 classes) use karte hain "
         "as a public benchmark. Healthcare to motivating example hai &mdash; "
         "techniques same hi rahengi."),
    ]
    for q, a in qa:
        P(f"<b>{q}</b>", "h2")
        P(a)

    flow.append(PageBreak())

    # ---------- FILE MAP ----------
    P("9. File-by-file map", "h1")

    files = [
        ("Code/dashboard.py", "Streamlit 10-page dashboard (~2,061 lines). UI entry point."),
        ("Code/run_demo.py", "Phase 1 + Phase 2 ek shot mein. Plots banata hai."),
        ("Code/src/models/moe_model.py", "LoRA, MoEExpert, MoELayer, MoETextClassifier."),
        ("Code/src/data/text_datasets.py", "AG News loader, vocab build, IID + Dirichlet split."),
        ("Code/src/fl/client.py", "local_train(): forward+backward, FedProx option, sparse pack."),
        ("Code/src/fl/server.py", "FedServer: aggregate, aggregate_median, aggregate_trimmed_mean, "
                                "aggregate_secure, aggregate_with_verification."),
        ("Code/src/fl/dp.py", "clip_update, add_noise, apply_dp, PrivacyAccountant, RenyiAccountant."),
        ("Code/src/fl/sepg.py", "SEPGProof, generate_proof, verify_proof, oracle_score, "
                              "committee_decision."),
        ("Code/src/fl/adversaries.py", "poisoning_train, freerider_train, sybil_clones."),
        ("Code/experiments/run_all_experiments.py", "4 experiments &rarr; PNGs + JSON in plots/."),
        ("Code/data/ag_news_train.csv", "120,000 train samples (label, text)."),
        ("Code/data/ag_news_test.csv", "7,600 test samples."),
        ("Code/plots/", "Pre-computed PNGs + experiment_results.json."),
        ("Report/main.tex", "Final report LaTeX (46 references, full methodology)."),
    ]
    fm = [["File / Folder", "Kya hai"]]
    for a, b in files:
        fm.append([Paragraph(f'<font face="Courier" size="8">{a}</font>', st["body"]),
                   Paragraph(b, st["body"])])
    fmt = Table(fm, colWidths=[6.5*cm, 9.5*cm])
    fmt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d7c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.HexColor("#f4f7fb"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(fmt)

    SP(14)

    # ---------- TEAM / FINAL ----------
    P("10. Team", "h1")
    P(
        "<b>Group #34 &mdash; IIIT Kota</b><br/>"
        "&bull; Keshav Kashyap (2023KUCP1161)<br/>"
        "&bull; Lakshya Sharma (2023KUCP1167)<br/>"
        "&bull; Prakriti Patel (2023KUCP1109)<br/>"
        "<b>Supervisor:</b> Dr. Gyan Singh Yadav, CSE Dept., IIIT Kota"
    )

    SP(10)
    P("Quick recap (1 line each)", "h1")
    recap = [
        "<b>Kya hai:</b> Privacy + Verifiable + Communication-efficient Federated Learning.",
        "<b>Data:</b> AG News, 120K headlines, 4 classes, 5 simulated clients.",
        "<b>Privacy:</b> DP-SGD (clip + Gaussian noise) + Renyi accountant.",
        "<b>Verifiability:</b> SEPG proof &mdash; SHA-256 + 4 structural checks.",
        "<b>Comm savings:</b> Top-K MoE &rArr; ~75% bandwidth reduction.",
        "<b>Robustness:</b> Median agg &rArr; -7pp drop at 40% malicious (vs FedAvg -16pp).",
        "<b>Run:</b> python -m streamlit run dashboard.py &rArr; localhost:8501.",
    ]
    for r in recap:
        P(f"&bull; {r}", "bullet")

    SP(20)
    P("&mdash; END &mdash;", "subtitle")

    doc.build(flow)
    print(f"PDF generated: {OUT}")
    print(f"Size: {OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
