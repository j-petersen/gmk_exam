import os
import numpy as np
import pandas as pd
import typhon as ty
import matplotlib.pyplot as plt

GRADE_LABELS = [
    "5.0",
    "4.0",
    "3.7",
    "3.3",
    "3.0",
    "2.7",
    "2.3",
    "2.0",
    "1.7",
    "1.3",
    "1.0",
]


def main():
    filename = "GMK-Abschlussklausur-Bewertungen.csv"

    ty.plots.styles.use(["typhon"])
    df = readin_exam_data(filename)

    # test different grading systems
    grading_steps = [
        create_grading_array(50, 100),
        create_grading_array(45, 90),
        create_grading_array(42, 72),
        create_grading_array(40, 90),
        create_grading_array(36, 76),
        create_grading_array(30, 80),
        create_grading_array(35, 85),
    ]

    for grading in grading_steps:
        plot_notenspiegel(df, [grading])

    plot_question_procentage(df)
    plot_point_percentile(df, grading_steps)

    tuhh_df = get_students_with_str(df, str="tuhh")
    uhh_df = get_students_with_str(df, str="uni-hamburg")
    plot_notenspiegel_tuhh_vs_uhh(tuhh_df, uhh_df, create_grading_array(35, 85))
    plot_notenspiegel(df, [create_grading_array(35, 85)])

    # write final results
    write_final_df(
        tuhh_df, create_grading_array(35, 85), "GMK_exam_tuhh_correction.csv"
    )
    write_final_df(uhh_df, create_grading_array(35, 85), "GMK_exam_uhh_correction.csv")

    plt.show()


def plot_point_percentile(df, grading_steps=None):
    """plot the percentiles to see how skew the distribution is."""
    if grading_steps is None:
        grading_steps = [create_grading_array(50, 100)]

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    for i in range(len(grading_steps)):
        p = grading_steps[i]
        percentiles = np.nanpercentile(df[f"Percent"], p)

        ax.plot(p, percentiles, linestyle=None, marker="x", label=f"{p[1]}-{p[-2]}")

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Erreichte Prozent")
    ax.set_xticks(p)
    ax.set_ylim(0, 100)
    ax.legend()

    ax.grid(True)
    ax.set_axisbelow(True)

    fig.savefig(os.path.join("plots", "percentile_exam.png"), dpi=180)


def plot_question_procentage(df):
    """plot the achived procentage of all participants per proplem."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    gesamtschnitt = np.array(
        df[df["Nachname"] == "Gesamtdurchschnitt"].values.tolist()[0][10:]
    )
    max_points = np.array(
        df[df["Nachname"] == "Mögliche Anzahl"].values.tolist()[0][10:]
    )

    ax.bar(list(df.columns[10:]), np.divide(gesamtschnitt, max_points) * 100)
    ax.set_xlabel("Frage")
    ax.set_ylabel("Erreichte Prozent")
    ax.set_xticks(np.arange(0, 60, 3))
    ax.set_ylim(0, 100)

    ax.grid(True)
    ax.set_axisbelow(True)

    fig.savefig(os.path.join("plots", "punkte_pro_aufgabe_exam.png"), dpi=180)


def plot_notenspiegel(df, grading_steps=None):
    """plot the grades of all participants in the dataframe."""
    if grading_steps is None:
        grading_steps = [create_grading_array(50, 100)]

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    bar_width = 0.8 / len(grading_steps)

    for i in range(len(grading_steps)):
        h, _ = np.histogram(df[f"Percent"], bins=grading_steps[i])

        label = f"{grading_steps[i][1]}-{grading_steps[i][-2]}"
        if len(grading_steps) != 1:
            bar_x = np.array(range(len(GRADE_LABELS))) - 0.4 + bar_width * i
        else:
            bar_x = range(len(GRADE_LABELS))

        rect = ax.bar(bar_x, h, width=bar_width, edgecolor="k", label=label)
        # ax.bar_label(rect, padding=3)

        Notendurchschnitt = calc_average_grade(h)

        print(
            "Der Notendurchschnitt ({}) liegt bei: {:.2f}".format(
                label, Notendurchschnitt
            )
        )

    if len(grading_steps) == 1:
        add_grade_ranges(ax, grading_steps)
        ax.text(
            -0.5, np.max(h), "Schnitt: {:.2f}".format(Notendurchschnitt), fontsize=24
        )

    ax.set_xlabel("Note")
    ax.set_ylabel("Anzahl")

    ax.set_xticks(range(len(GRADE_LABELS)))
    ax.set_xticklabels(GRADE_LABELS)

    ax.grid(True)
    ax.set_axisbelow(True)

    if len(grading_steps) != 1:
        ax.legend()

    if len(grading_steps) == 1:
        fig.savefig(
            os.path.join(
                "plots", f"notenspiegel{grading_steps[0][1]}-{grading_steps[0][-2]}.png"
            ),
            dpi=180,
        )
    else:
        fig.savefig(os.path.join("plots", "notenspiegel_exam.png"), dpi=180)


def plot_notenspiegel_tuhh_vs_uhh(df_tuhh, df_uhh, grading_steps=None):
    """Plot the gradings split between tuhh and uhh students."""
    if grading_steps is None:
        grading_steps = create_grading_array(50, 100)

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    h_tuhh, _ = np.histogram(df_tuhh[f"Percent"], bins=grading_steps)
    h_uhh, _ = np.histogram(df_uhh[f"Percent"], bins=grading_steps)

    rect = ax.bar(
        range(len(GRADE_LABELS)),
        h_tuhh,
        width=0.8,
        color="lightblue",
        edgecolor="k",
        label="tuhh (n={:.0f}, mean={:.2f})".format(
            h_tuhh.sum(), calc_average_grade(h_tuhh)
        ),
    )
    rect = ax.bar(
        range(len(GRADE_LABELS)),
        h_uhh,
        width=0.8,
        bottom=h_tuhh,
        color="red",
        edgecolor="k",
        label="uhh (n={:.0f}, mean={:.2f})".format(
            h_uhh.sum(), calc_average_grade(h_uhh)
        ),
    )

    ax.set_xlabel("Note")
    ax.set_ylabel("Anzahl")

    ax.set_xticks(range(len(GRADE_LABELS)))
    ax.set_xticklabels(GRADE_LABELS)

    ax.grid(True)
    ax.set_axisbelow(True)

    ax.legend()

    fig.savefig(os.path.join("plots", "notenspiegel_tuhh_uhh_exam.png"), dpi=180)


def add_grade_ranges(ax, grading_steps):
    """Add the percantages to achive the respective grade to the plot."""
    for i in range(len(grading_steps[0]) - 1):
        grading_range = "{:.0f}-{:.0f}".format(
            grading_steps[0][i], grading_steps[0][i + 1]
        )
        ax.text(i - 0.3, 0.2, grading_range, fontsize=16)


def calc_average_grade(hist):
    """Calculate the average grade."""
    float_grades = np.array([5.0, 4.0, 3.7, 3.3, 3.0, 2.7, 2.3, 2.0, 1.7, 1.3, 1.0])
    average = (np.array(hist) * float_grades).sum() / hist.sum()
    return average


def create_grading_array(lower_bound, upper_bound):
    """Create a grading systems from the `lower_bound` to the `upper_bound`."""
    grading = np.linspace(lower_bound, upper_bound, 11)
    if grading[0] != 0:
        grading = np.insert(grading, 0, 0)
    if grading[-1] != 100:
        grading[-1] = 100
    return grading


def readin_exam_data(filename):
    """Read in the data file with the moodle format."""
    filedir = "data"
    df = pd.read_csv(os.path.join(filedir, filename), na_values="-", decimal=",")
    column_names = list(df.columns)
    questions = list(filter(lambda ele: ele.startswith("F"), column_names))

    # add row with max points for each question
    df = df.append(
        dict(
            zip(
                ["Nachname"] + questions,
                ["Mögliche Anzahl"]
                + [float(ele.split("/")[-1].replace(",", ".")) for ele in questions],
            )
        ),
        ignore_index=True,
    )

    # remove a question if something is wrong (to hard or false solution)
    # note that the question number here is (for some unclear reason) one higher as the moodle question number
    df = remove_question(df, 19)

    # rename question columns
    df = df.rename(
        columns=dict(
            zip(questions, [str(int(ele.split("/")[0][1:])) for ele in questions])
        )
    )

    total_points = df.iloc[-1, 9:].sum()
    df["Percent"] = df.iloc[:, 9:].sum(axis=1) / total_points * 100

    print(f"Gesamtpunkte: {total_points}")

    # test if total sum of all points is 100%
    assert (
        df["Percent"].iloc[-1] == 100
    ), "Achtung! Irgendwas läuft schief. Die Summe aller Punkte ist nicht erreicht nicht 100%. Check, ob die Anzahl der Columns falsch gesetzt ist in der Einleseroutine."

    return df


def remove_question(df, question_number):
    """Sets all points to nan. Basically disregards the question."""
    df[df.columns[df.columns.str.contains(str(question_number))]] = np.NAN
    return df


def write_final_df(df, grading_array, filename, filedir="data"):
    """write a .csv with the final grade and percentage of all participants in the dataframe."""
    df = add_grade_column(df, grading_array)
    df = df.drop(
        ["Status", "Begonnen am", "Beendet", "Verbrauchte Zeit", "Bewertung/97,00"],
        axis=1,
    )
    df = df.drop([str(num) for num in range(1, 60)], axis=1)
    df["Percent"] = np.round(df["Percent"], 2)
    df.to_csv(os.path.join(filedir, filename), index=False)


def get_students_with_str(df, str="tuhh"):
    """Create sub dataframe by searching for a string in the mail address of the participants.
    For example, it can be used to get all TUHH students."""
    df = df.dropna(subset=["E-Mail-Adresse"])
    contains_df = df[df["E-Mail-Adresse"].str.contains(str)]
    # discludes_df = df[str not in df['E-Mail-Adresse']]
    return contains_df


def add_grade_column(df, grading_array):
    """Add a column to the ddataframe with the according grade."""
    df["Note"] = calc_grade(grading_array, df[f"Percent"].values)
    return df


def calc_grade(grading_array, percentage):
    """From percentage detemine the grade according to the grading_array."""
    grade_labels = [
        "5.0",
        "4.0",
        "3.7",
        "3.3",
        "3.0",
        "2.7",
        "2.3",
        "2.0",
        "1.7",
        "1.3",
        "1.0",
    ]
    grades = []
    for percent in percentage:
        grades.append(grade_labels[ceilSearch(grading_array, percent)])
    return grades


def ceilSearch(arr, x):
    """Ceilling Search adjusted to the grading system!"""
    if x < arr[0]:
        return 0
    for i in range(len(arr) - 1):
        # found the number
        if arr[i] == x:
            return i
        # number between ele i and ele i+1
        if arr[i] < x and arr[i + 1] >= x:
            return i
    # x greater then last ele in array
    return len(arr) - 1


if __name__ == "__main__":
    main()
