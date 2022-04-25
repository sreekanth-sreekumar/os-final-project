import environment
from scheduler import ComparisonClass

if __name__ == '__main__':
    env, scheduler = environment.set_up_enviroment(load_environment=True, load_scheduler=True)

    while not env.terminated():
        actions = scheduler.schedule()
        print(actions)
    print('\n\nEND')

    comparison_class = ComparisonClass(env)
    comparison_class.comparison()