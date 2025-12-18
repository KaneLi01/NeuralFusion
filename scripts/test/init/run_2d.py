from neural_fusion.pipelines.projective import ProjectivePipeline

if __name__ == "__main__":
    pjp = ProjectivePipeline()
    points = pjp.get_data()
    print(points)
