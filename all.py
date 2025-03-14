6 th class and objects :

class Student:
    def __init__(self, name, age, marks):
        self.name = name
        self.age = age
        self.marks = marks

    def calculate_average_score(self):
        if len(self.marks) > 0:
            return sum(self.marks) / len(self.marks)
        else:
            return 0

    def calculate_result(self):
        average_score = self.calculate_average_score()
        if average_score < 50:
            return "Fail"
        elif average_score < 60:
            return "Second Class"
        elif average_score < 90:
            return "First Class"
        else:
            return "Distinction"

# Create an empty list to store student objects
students = []

# Collect information for multiple students in a loop
num_students = int(input("Enter the number of students: "))

for i in range(num_students):
    name = input(f"Enter the name of student {i + 1}: ")
    age = int(input(f"Enter the age of student {i + 1}: "))
    

    marks = []
    for j in range(2):
        subject_mark = float(input(f"Enter the mark for subject {j + 1} for student {i + 1}: "))
        marks.append(subject_mark)

    # Create a Student object and add it to the list
    student = Student(name, age, marks)
    students.append(student)

# Analyze and display student information, including results
print("\nResult(s):")
for i, student in enumerate(students):
    print(f"Student {i + 1}:")
    print(f"Name: {student.name}")
    print(f"Age: {student.age}")
    print(f"Average Score: {student.calculate_average_score()}")
    result = student.calculate_result()
    print(f"Result: {result}")
------------------------------------------------
7th program Inheritance :
class StudentDetails:
    def __init__(self, name, student_id, programme):
        self.name = name
        self.student_id = student_id
        self.programme = programme  # Store the programme info

    def student_info(self):
        return f"Student ID: {self.student_id}\nName: {self.name}\nProgramme: {self.programme}"


# Class to handle academic details like marks and grades
class Student:
    def __init__(self, student_details):
        self.student_details = student_details  # Composition: Student has a StudentDetails object
        self.marks = {}

    def add_marks(self, subject, marks):
        self.marks[subject] = marks
        # Removed the print statement here

    def calculate_grade(self):
        total_marks = sum(self.marks.values())
        num_subjects = len(self.marks)

        if num_subjects == 0:
            return "No marks available"
        
        percentage = total_marks / num_subjects
        if percentage >= 75:
            return "A"
        elif percentage >= 60:
            return "B"
        elif percentage >= 50:
            return "C"
        elif percentage >= 40:
            return "D"
        else:
            return "F"

    def student_info(self):
        # Get student's personal details from the StudentDetails object
        personal_info = self.student_details.student_info()
        # Add academic details like marks and grade
        return f"{personal_info}\nMarks: {self.marks}\nGrade: {self.calculate_grade()}"


# Main function to gather input and create student instances
def main():
    # Collect Student Details
    print("Enter Student Details:")
    name = input("Enter Name: ")
    student_id = input("Enter Student ID: ")
    programme = input("Enter Programme (UG/PG): ")  # Ask user for programme (UG or PG)
    
    student_details = StudentDetails(name, student_id, programme)
    
    # Create Student instance for academic details
    student = Student(student_details)
    
    # Get marks for the student
    num_courses = int(input("How many courses' marks should be entered?: "))
    for i in range(1, num_courses + 1):
        marks = float(input(f"Enter the course {i} mark: "))  # Prompt for each course mark
        subject = f"Course {i}"  # Automatically name the subject as "Course X"
        student.add_marks(subject, marks)
    
    # Display student information
    print("\nStudent Academic Information")
    print("----------------------------")
    print(student.student_info())  # Display complete student info


if __name__ == "__main__":
    main()




------------------------------------------------

8th exception handling:

class Contact:
    def __init__(self, name, phone_number, email):
        self.name = name
        self.phone_number = phone_number
        self.email = email

    def __str__(self):
        return f"Name: {self.name}, Phone: {self.phone_number}, Email: {self.email}"


class ContactManager:
    def __init__(self):
        self.contacts = []

    def add_contact(self, name, phone_number, email):      
        contact = Contact(name, phone_number, email)
        self.contacts.append(contact)
        print(f"Contact '{name}' added successfully.")

    def search_contact(self, name):
        for contact in self.contacts:
            if contact.name.lower() == name.lower():
                return contact
        return None

    def display_contacts(self):
        if not self.contacts:
            print("No contacts available.")
            return
        print("\nContact List:")
        for contact in self.contacts:
            print(contact)


def main():
    manager = ContactManager()
    while True:
        print("\nContact Management Menu:")
        print("1. Add Contact")
        print("2. Search Contact")
        print("3. Display All Contacts")
        print("4. Exit")

        choice = input("Choose an option (1-4): ")
        try:
            if choice == '1':
                name = input("Enter the contact's name: ")
                phone_number = input("Enter the contact's phone number: ")
                email = input("Enter the contact's email: ")
                manager.add_contact(name, phone_number, email)
            elif choice == '2':
                name = input("Enter the name of the contact to search: ")
                contact = manager.search_contact(name)
                if contact:
                    print(contact)
                else:
                    print(f"Contact '{name}' not found.")
            elif choice == '3':
                manager.display_contacts()
            elif choice == '4':
                print("Exiting the contact management system.")
                break
            else:
                print("Invalid choice. Please select a valid option.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
------------------------------------------------
9 th matplot lib :

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(iris.head())

# Basic Plotting: Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species', style='species', s=100)
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.grid()
plt.show()

# Customization of Plots: Line Plot (average sepal width by species)
avg_sepal_width = iris.groupby('species')['sepal_width'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(avg_sepal_width['species'], avg_sepal_width['sepal_width'], marker='o', linestyle='-', color='b')
plt.title('Average Sepal Width by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Width (cm)')
plt.grid()
plt.show()

# Subplots: Petal Length and Width
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(ax=axes[0], data=iris, x='species', y='petal_length', hue='species', palette='pastel', legend=False)
axes[0].set_title('Boxplot of Petal Length by Species')

sns.boxplot(ax=axes[1], data=iris, x='species', y='petal_width', hue='species', palette='pastel', legend=False)
axes[1].set_title('Boxplot of Petal Width by Species')

plt.tight_layout()
plt.show()

# Bar Plot: Count of Species
plt.figure(figsize=(8, 5))
sns.countplot(data=iris, x='species', hue='species', palette='Set2', legend=False)
plt.title('Count of Iris Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Histogram: Distribution of Sepal Length
plt.figure(figsize=(8, 5))
plt.hist(iris['sepal_length'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Pie Chart: Species Distribution
species_counts = iris['species'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Distribution of Iris Species')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Save a Figure
plt.figure(figsize=(8, 5))
sns.countplot(data=iris, x='species', hue='species', palette='Set2', legend=False)
plt.title('Count of Iris Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.savefig('iris_species_count.png')
plt.show()

------------------------------------------------
10 th Seaborn :

import seaborn as sns
# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(iris.head())

# Pairplot to visualize relationships between all features
plt.figure(figsize=(10, 6))
sns.pairplot(iris, hue='species', markers=["o", "s", "D"])
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Heatmap to visualize the correlation between features
plt.figure(figsize=(8, 6))
correlation = iris.corr(numeric_only=True)  # Only take numeric columns
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Iris Features')
plt.show()

# Violin plot for petal length and width by species
plt.figure(figsize=(10, 6))
sns.violinplot(data=iris, x='species', y='petal_length', hue='species', inner='quartile', legend=False)
plt.title('Violin Plot of Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()


# Strip plot to show distribution of petal width by species
plt.figure(figsize=(10, 6))
sns.stripplot(data=iris, x='species', y='petal_width', jitter=True, hue='species', legend=False)
plt.title('Strip Plot of Petal Width by Species')
plt.xlabel('Species')
plt.ylabel('Petal Width (cm)')
plt.show()

------------------------------------------------
