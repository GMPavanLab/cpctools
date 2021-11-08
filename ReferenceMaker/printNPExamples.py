from . import dhfat3049
from . import ico5083
from . import th4116
import numpy

"""
for np in (dhfat3049, ico5083, th4116):
    npName=''
    arrays=[]
    for element in [a for a in dir(np) if "__" not in a and a != "array"]:
        print (element)
"""


def _xyzExample(name, atoms, properties):
    nAtoms = len(atoms)
    atoms *= 1.44
    prop = "Properties=species:S:1:pos:R:3"
    data = []
    for p in properties:
        prop += f':{p["name"]}:I:1'
        d = numpy.zeros((nAtoms), dtype=numpy.int32)
        for index in p["indexes"]:
            d[index] = 1
        data.append(d)

    with open(f"{name}.xyz", "w") as oufile:
        oufile.write(f"{nAtoms}\n")
        oufile.write(f"{prop}\n")
        for i in range(nAtoms):
            oufile.write(f"Au {atoms[i,0]} {atoms[i,1]} {atoms[i,2]}")
            for p in data:
                oufile.write(f" {p[i]}")
            oufile.write("\n")


def printNPExamples():
    # {'name':"",'indexes':}
    _xyzExample(
        "dhfat3049",
        dhfat3049.dhfat3049,
        [
            {"name": "Concave_dh", "indexes": dhfat3049.maskConcave_dh},
            {"name": "FiveFoldedAxis_dh", "indexes": dhfat3049.maskFiveFoldedAxis_dh},
        ],
    )
    _xyzExample(
        "ico5083",
        ico5083.ico5083,
        [
            {"name": "Edges_ico", "indexes": ico5083.maskEdges_ico},
            {"name": "Face111_ico", "indexes": ico5083.maskFace111_ico},
            {"name": "FiveFoldedAxis_ico", "indexes": ico5083.maskFiveFoldedAxis_ico},
            {"name": "Vertexes_ico", "indexes": ico5083.maskVertexes_ico},
        ],
    )

    _xyzExample(
        "th4116",
        th4116.th4116,
        [
            {"name": "Edges_th", "indexes": th4116.maskEdges_th},
            {"name": "Face001_th", "indexes": th4116.maskFace001_th},
            {"name": "Face111_th", "indexes": th4116.maskFace111_th},
            {"name": "Vertexes_th", "indexes": th4116.maskVertexes_th},
        ],
    )


if __name__ == "__main__":
    printNPExamples()
