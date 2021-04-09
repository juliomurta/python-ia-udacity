names =  input("Enter names separated by commas: ")
assignments =  input("Enter assignments separated by commas: ")
grades =  input("Enter grades separated by commas: ")

nam_list = names.split(",")
asg_list = assignments.split(",")
grd_list = grades.split(",")

for name, assignment, grade in zip(nam_list, asg_list, grd_list):
    message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
    submit before you can graduate. You're current grade is {} and can increase \
    to {} if you submit all assignments before the due date..\n\n".format(name, assignment, grade, int(grade) + int(assignment)*2)

    print(message)