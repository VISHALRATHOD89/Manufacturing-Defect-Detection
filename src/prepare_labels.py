import pandas as pd

def build_image_labels(csv_path="data/severstal/train.csv"):
    df = pd.read_csv(csv_path)

    print("Raw train.csv shape:", df.shape)
    print(df.head())
    print("\nUnique ClassId values:", df["ClassId"].unique())
    print("Unique images with at least one defect:", df["ImageId"].nunique())

    return df

def pivot_to_image_labels(df):
    df["has_defect"] = 1

    pivoted = df.pivot_table(
        index="ImageId",
        columns="ClassId",
        values="has_defect",
        fill_value=0
    )

    pivoted.columns = [f"defect_{c}" for c in pivoted.columns]
    pivoted = pivoted.reset_index()

    return pivoted    


def add_clean_images(labels_df, train_images_dir="data/severstal/train_images"):
    import os

    all_images = set(os.listdir(train_images_dir))
    labeled_images = set(labels_df["ImageId"])
    clean_images = all_images - labeled_images

    print(f"\nTotal images: {len(all_images)}")
    print(f"Images with defects: {len(labeled_images)}")
    print(f"Clean images (no defect): {len(clean_images)}")

    clean_df = pd.DataFrame({
        "ImageId": list(clean_images),
        "defect_1": 0, "defect_2": 0, "defect_3": 0, "defect_4": 0
    })

    full_df = pd.concat([labels_df, clean_df], ignore_index=True)
    return full_df



if __name__ == "__main__":
    df = build_image_labels()
    labels_df = pivot_to_image_labels(df)

    full_labels_df = add_clean_images(labels_df)

    print("\nFull label table shape:", full_labels_df.shape)
    print(full_labels_df["defect_1"].value_counts())

    full_labels_df.to_csv("data/severstal/image_labels.csv", index=False)
    print("\nSaved -> data/severstal/image_labels.csv")