from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate


class Chaptering:

    def __init__(self, text):
        self.text = text

    def get_chapters(self):
        llm = GoogleGenerativeAI(model="models/text-bison-001",
                                 google_api_key='AIzaSyCQgdRO8gDEpr-DUCqaCi0LOE_T4_Ns1OU',
                                 temperature=0.7)
        prompt_template = PromptTemplate(
            input_variables=["segment", "output_format"],
            template="""Generate the chapters title for the following transcript: {segment} just give the titles no additional things required and display in the output format for all the chapters as mentioned here
                        Output Format:
                        {output_format}

                        Example Output:

                        Chatper No.1 - Chapter Title 1
                        Chatper No.1 - Chapter Title 2
                        Chatper No.1 - Chapter Title 3
                        Please ensure that the output strictly adheres to the format provided, with no additional information beyond the chapter titles.
                        """
        )
        chain = prompt_template | llm
        output_format = ("""
                            <li> chapter no - chapter title </li>
                         """)
        response = chain.invoke({"segment": self.text, "output_format": output_format})
        return response


# text = "Artificial Intelligence is a huge opportunity right now and combined with high quality German education, that's an absolute monster. So in this video I'm going to tell you everything you need to know about studying Artificial Intelligence in Germany. The requirements, career opportunities and salaries. And at the end of the video I'll present you the best universities in Germany for Artificial Intelligence bachelors and masters. The digital transformation is happening fast right now and there's a crazy crazy boom in AI. Since the demand is so high, there's a high chance that there will be a shortage of specialists in all industries. And I believe this is why more and more universities in Germany are starting to offer brand new courses in AI. So far there aren't many courses in Germany. Right now you can study it at 83 universities in 153 degree programs. So there's one course dedicated to studying Artificial Intelligence in Germany that is really dedicated to studying in two ways. Germany. Right now you can study it at 83 universities in 153 degree programs. So there are two ways to study Artificial Intelligence in Germany. One is to study in a dedicated course that really specializes in Artificial Intelligence. The course usually has the name AI in it. A good example is the bachelor's degree at Ludwig Maximilian University in Munich. This one is a form of applied computer science. The second way is to enroll in a computer science related course and learn about AI as part of the curriculum. You might have a single module, a two-semester seminar or a specialization subject. For example, I study computer science and design and our topic this semester is Artificial Intelligence. So we have a couple of courses about it. Other degrees that contain a lot are robotics, machine learning, data and business analytics. It takes a semester to get this sixth standard science degree of AI and if you want to do that, you need to do that sixth standard science degree. Machine learning, data science and business analytics. It takes a standard six to seven semesters to get this Bachelor of Science degree. And if you want to do a master's degree after that, you can follow up with a three to four semester course. Here you will have much more opportunities to specialize and write your big boy master's thesis in the final semester. In addition to these two degrees, there are also different study formats. Long distance study or a dual program in artificial intelligence may also be a good fit for you if practical application is your thing. I have made a dedicated video about dual studies. Check it out after watching this video. Now what are the requirements and skills you need to study artificial intelligence? The exact requirements are of course different from university to university because often is an interdisciplinary degree program. So it is offered in combination with another field depending on whether you want to study artificial intelligence. From university to university because often artificial intelligence is an interdisciplinary degree program. So it is offered in combination with another field. Depending on whether you want to study at a classic university or a university of applied sciences, you will need a school leaving certificate. Some courses are also open to master craftsmen and qualified professionals. So people who have completed vocational training and have enough work experience. For example, if you look at the degree program artificial intelligence at LMU, this one starts in the winter semester and is open admission. So there is no minimum grade you need to get accepted. The language of instruction is German. So as an international student, you need at least B2 to enroll. Now all that focus on artificial intelligence courses have one thing in common. You can't avoid. An unwritten requirement is good math science and math related basic skills and math related degrees. One major reason is that students underestimate the importance of mathematics. In the first couple of semesters, many students get overwhelmed by the workload of mathematics and quit. Especially statistics is important. That's what I realized this semester since we have a lot of AI heavy modules. If you don't like complex math and coding, though, there are still ways to approach AI research from a different angle. You may not become an AI developer, but you can work with them. Companies and research institutes also have people from other fields like business, chemistry, or psychology. Now the content of each university course is a bit different because this course focuses on a different area of AI content. The content of the AI courses at each university is of course a little different. This is because each university focuses on a different area of study. The AI degree at TH Deggendorf is different from that at Lübeck University for example. Before you make your decision, you should definitely take a look at the study plans. Each degree program has a large PDF that explains all the modules in detail. The infamous Modulhandbuch. Check it out, it's really important. But I would say most of the AI courses are going to cover the basic modules here. The most important programming languages for AI development are Python and R, so that you can do data analysis, statistics, and write bachelor algorithms. But you don't need coding experience to degree in an AI It's, so don't worry. I have a bachelor's degree in AI, so don't worry. It will really help to know some programming for example as part of your hobby, but you will learn everything you need during your studies. Before I started studying computer science and design, I didn't really have coding experience, Now study programs related to AI cover a wide range of topics. There are technical aspects, but on top of that also legal, ethical, and social issues. So next to coding and computer science, you will automatically learn about math, neurology, psychology, and ethics. This is really powerful in my eyes because you're going to be multidimensional skilled. Some degree programs are also more practical than others. Especially at universities of applied sciences, you can work on dedicated projects and do internships at companies. Now studying at public universities AI is generally free of charge. Now, studying AI at public universities is generally free. A couple of universities charge tuition fees for international students, though, The Technical University of Munich, where you can study AI as part of their robotics program, has recently introduced tuition fees for example. Now there are also quite a lot of private universities that offer AI courses. I believe this is because this type of degree program is very new and private universities are faster when it comes to launching new innovative degree programs compared to classical universities. Tuition fees are quite heavy at private universities. It's usually a couple thousand euros per semester. I personally have no experience with private ones, so you have to decide for yourself if such an amount is worth it. But what I can say is that the public ones are enough and will give you absolutely everything you need to know for free. What I can tell you is that the career in AI is extremely paying off. But what I can say is that the public ones are absolutely enough and will give you everything you need to know for free. What I can tell you is that the career opportunities in AI are extremely good. As a computer science student myself, I was also curious what the opportunities and salaries are. So I connected with some individuals in this industry and yes, they are all very satisfied with their career progression and income. I think students are recognizing that if you have a background in AI, so many doors to interesting and well-paid jobs are open. Companies in Germany and around the world are increasingly looking for AI specialists and people have realized that there is an extreme shortage of talent in this field. Artificial intelligence developers are certainly among the highest earning computer scientists and the starting salary as a data science and machine learning employee is around 60,000 euros per year and early AI developers can expect to earn six figures in this field. The best thing is that you can work in so many different industries. Practical experience is even more important for companies than a master's degree. You can also see this in job offers. Many companies want you to have a couple of years of experience in the field of software development. I can really recommend application-oriented degrees and dual study programs here to get a lot of exposure to the job industry early. I believe in the coming years, a lot of brand new AI jobs will appear on the job market and to prepare for that, it's good to figure out which universities offer this kind of degree program. So if you want to figure out what the best universities for an AI degree are, we have to look at the industries and companies located around these universities offer this kind of degree program. So if you want to figure out what the best universities for an AI degree are, we have to look at the industries and companies located around these institutions because this whole AI development thing is often done in cooperation with industry partners. And we want to have plenty of career opportunities after graduation, right? I would say Munich is one of the biggest tech hubs in Germany right now with major tech companies like IBM, Microsoft and Google. Then there's the thing called Cyber Valley in Germany. It's one of the biggest AI research associations in Europe with big names like Max Planck and Fraunhofer Institute, Mercedes Benz Group, BMW, Porsche and Amazon. Big names. The University of Tübingen and the University of Stuttgart are part of this and offer bachelor's and master's degrees in AI I've chosen 10 programs for almost every major university in Germany. Now you can take a screenshot of this AI if you want to. I have selected 10 degree programs in AI for you. You can screenshot this if you want. Now almost every major university in Germany offers programs in data science and computer science. There are multiple ways to get into the AI industry. Don't forget to enroll in the free Germany Starter Kit course in the video description. Download the free study in Germany guidebook and join our huge Discord community. Love you and stay focused. Now, the Technical University of Munich offers a master's program in robotics, cognition and intelligence, and data engineering and analytics. There is also a course about AI and society, which is more about regulations and innovation management. And at the University of Applied Sciences in Munich, you can study three AI-related degree programs as part of the Munich Center for Digital Sciences and AI. I'm enrolled in one of them, computer science and design."
# chaptering = Chaptering(text)
# chapters = chaptering.get_chapters()
# print(chapters)