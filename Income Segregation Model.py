import random
import numpy as np
import math
import statistics
import tkinter as tk
import matplotlib


class Display(tk.Frame):  # Inherit from Frame

    def __init__(self, sim, g_s, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.sim = sim
        self.grid_size = g_s
        self.state = 0
        self.screen_size = [640, 640] # Modify as needed to fit screen

        self.alpha_x = self.screen_size[0]/self.grid_size[0]
        self.alpha_y = self.screen_size[1]/self.grid_size[1]

        self.city_canvas = tk.Canvas(self, bd=0, bg="black", width=self.screen_size[0], height=self.screen_size[1])
        self.next_period = tk.Button(self, text="Next Period", command=self.updateFrame)
        self.period_label = tk.Label(self, text="Period: "+"%s" % self.sim.period, font=('Helvetica', 12))
        self.max_income = max(agent.income for agent in self.sim.agents)
        self.median_income = statistics.median(agent.income for agent in self.sim.agents)
        self.gini_label = tk.Label(self, text="Gini Coefficient: "+"%s" % Calculation.calculateGini(Calculation(), self.sim), font=('Helvetica', 10))
        self.moran_label = tk.Label(self, text="Moran's I: " + "%s" % Calculation.calculateMoranI(Calculation(), self.sim), font=('Helvetica', 10))
        self.updateFrame()

        self.state = 1

        self.period_label.pack()
        self.gini_label.pack()
        self.moran_label.pack()
        self.city_canvas.pack()
        self.next_period.pack()




    def updateFrame(self):
        if self.state == 1:
            self.sim.nextPeriod()
            self.period_label.config(text="Period: "+"%x" % self.sim.period)
            self.moran_label.config(text="Moran's I: " + "%s" % Calculation.calculateMoranI(Calculation(), self.sim))

        self.city_canvas.delete("all")
        for h, a in self.sim.ledger.items():
            h_colour = self.getColour(a)
            text_colour = "black"
            if h_colour[1] == 0 and h_colour[2] == 0:
                text_colour = "white"
            self.city_canvas.create_rectangle(h.location[0]*self.alpha_x+1, h.location[1]*self.alpha_y+1, (h.location[0]*self.alpha_x)+self.alpha_x+1, (h.location[1]*self.alpha_y)+self.alpha_y+1, fill='#%02x%02x%02x' % h_colour)
            self.city_canvas.create_text([h.location[0] * self.alpha_x + self.alpha_x / 2, (h.location[1] * self.alpha_y) + self.alpha_y / 2], text=a.ID, font=('Helvetica', 11), fill=text_colour)


    def getColour(self, a):

        proportion = a.income / self.max_income
        colour = [0, 0, 0]

        if proportion <= 0.5:
            colour[0] = (2*proportion)*255
            colour[1] = 0
            colour[2] = 0
        elif proportion > 0.5:
            colour[0] = 255
            colour[1] = (proportion-0.5)/0.5 * 255
            colour[2] = (proportion-0.5)/0.5 * 255


        rgb = (int(colour[0]), int(colour[1]), int(colour[2]))
        return rgb


class Agent(object):

    def __init__(self, ID, income, amenity_pref, mrs):
        self.ID = ID
        self.income = income
        self.amenity_pref = amenity_pref
        self.mrs = mrs  # Housing:Disposable income. The number of dollars toward housing that is equal in utility to 1 dollar of disposable income (0 = housing infinitely valuable).

    def bid(self, ledger):   # Crucial function. Bid made depends on utility and affordability.
        current_bids = []    # Array matching house location to the agent's bid on that location
        max_bid = 1/(1+self.mrs) * self.income  # The marginal propensity to consume housing * income

        houses = [house for house in ledger.keys()]
        houses_by_utility = sorted(houses, key=lambda house: house.base_utility)

        max_utility = houses_by_utility[-1].base_utility
        for house in houses_by_utility:
            bid = (house.base_utility/max_utility)*max_bid  # The bid is mrs between housing and disposable income * income * size of utility compared to maximum achievable utility in market
            current_bids.append([house, bid])

        # The output is of the form: [[<house object>, corresponding bid], [<house object>, corresponding bid], ...]
        return current_bids



class House(object):

    def __init__(self, location, neighbourhood):
        self.location = location
        self.neighbourhood = neighbourhood
        self.price = 0

        self.base_utility = 0


class Simulation(object):

    def __init__(self, n_a, i_r, g_s, n_p):
        self.n_agents = n_a
        self.income_range = i_r
        self.grid_size = g_s
        self.n_periods = n_p

        self.agents = []
        self.houses = []
        self.period = 0
        self.ledger = {}    # House:Agent Keeps track of ownership of each house. Every entity in self.houses and self.agents is matched based on ownership.

        self.main()

    def main(self):
        amenity_pref = 1

        for i in range(self.n_agents):
            self.agents.append(self.generateAgent(i+1, amenity_pref))
        self.houses = self.generateHouses() # A 2D array of instances of House() with index position [x][y] corresponding to house location.


        # Random assignment to houses. (Period 0)

        random.shuffle(self.agents)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for agent in self.agents:
                    if agent not in self.ledger.values():
                        self.ledger[self.houses[i][j]] = agent
                        break
                    else:
                        continue



    def generateAgent(self, ID, amenity_pref):
        # Normal distribution of incomes

        """
        random_incomes = list(np.random.normal(3000, 3000, size=(1000)))
        acceptable_incomes = []
        for i in range(len(random_incomes)):
            if random_incomes[i] <= 0 or random_incomes[i] >= 10000:
                continue
            else:
                acceptable_incomes.append(random_incomes[i])

        final_income = random.choice(acceptable_incomes)

        """

        # Random incomes

        """
        final_income = random.randint(self.income_range[0], self.income_range[1])
        """

        # Uniform distribution of incomes

        acceptable_incomes = [(i+1)*(self.income_range[1]-self.income_range[0])/400 for i in range(self.n_agents)]
        final_income = random.choice(acceptable_incomes)



        # Lognormal distribution of incomes based on US census data compiled by Schield
        """
        acceptable_incomes = list(np.random.lognormal(11.0302, 0.8179, size=1000))
        final_income = random.choice(acceptable_incomes)
        """

        # Choice of MRS

        acceptable_mrs = [1]
        final_mrs = random.choice(acceptable_mrs)


        new_agent = Agent(ID, final_income, amenity_pref, final_mrs)  # A uniform distribution of incomes.

        return new_agent

    def generateHouses(self):
        houses = [[0 for x in range(self.grid_size[0])] for y in range(self.grid_size[1])]  # Making a matrix the size of the grid

        # Determining the neighbours of each house
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                neighbourhood_pos = [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1], [x, y + 1], [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]
                neighbourhood = []

                for pos in neighbourhood_pos:
                    if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]:
                        neighbourhood.append(pos)

                new_house = House([x, y], neighbourhood)  # Assigning each house a location, starting at top left square as [0, 0], with a random price from a uniform distribution.
                houses[x][y] = new_house    # Populating matrix
        return houses

    def updateBaseUtilities(self):  # Call at any point to redetermine the base utilities of each house based on neighbours
        for house in self.ledger.keys():
            neighbour_utilities = []
            for pos in house.neighbourhood:
                neighbour = self.houses[pos[0]][pos[1]]
                neighbour_utilities.append(self.ledger[neighbour].income)

            house.base_utility = statistics.mean(neighbour_utilities)

    def generateAcceptances(self, current_bids, current_agents):
        acceptances = {}    # Format: {<agent>: [[house, max_bid], [house, max_bid], ...]

        for agent in current_agents:
            acceptances[agent] = []

        for house, bid_set in current_bids.items():
            max_bid_index = 0
            max_bid = max(bid_set[i][1] for i in range(len(bid_set)))
            for j in range(len(bid_set)):
                if bid_set[j][1] == max_bid:    # Picks the first bid it sees that is the maximum - deals with two bids of same max value (improbable but not impossible).
                    max_bid_index = j
                    break
                else:
                    continue

            acceptances[bid_set[max_bid_index][0]].append([house, max_bid])

        return acceptances


    def collateBids(self):
        all_bids = {}   # Format: {<house object>:[[<agent>, bid], [<agent>, bid]]}

        for row in self.houses:
            for house in row:
                all_bids[house] = []


        for agent in self.agents:
            bid_set = agent.bid(self.ledger)
            for bid in bid_set:
                all_bids[bid[0]].append([agent, bid[1]])

        return all_bids

    def nextPeriod(self):
        self.updateBaseUtilities()
        current_bids = self.collateBids()


        current_agents = [agent for agent in self.agents]
        new_ledger = {}
        movement = 0

        while len(current_agents) > 0:

            all_acceptances = self.generateAcceptances(current_bids, current_agents)
            for agent, acceptances in all_acceptances.items():
                if len(acceptances) > 0:
                    max_utility = max(acceptances[i][0].base_utility for i in range(len(acceptances)))
                    for j in range(len(acceptances)):
                        if acceptances[j][0].base_utility == max_utility:

                            new_ledger[acceptances[j][0]] = agent  # Transfer ownership

                            for house in current_bids.keys():  # Take house out of running
                                if house == acceptances[j][0]:
                                    current_bids.pop(house)
                                    break
                                else:
                                    continue

                            current_agents.remove(agent)  # Take agent out of running

                            # Take agent's bid out of running:
                            for house, bid_set in current_bids.items():
                                for bid in bid_set:
                                    if bid[0] == agent:
                                        bid_set.remove(bid)

                            break
                        else:
                            continue

            self.ledger = new_ledger
            movement += 1

        self.period += 1


class Calculation(object):

    def __init__(self):
        pass

    def calculateGini(self, sim):
        # Mean absolute difference

        x = [agent.income for agent in sim.agents]
        mad = np.abs(np.subtract.outer(x, x)).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        # Gini coefficient
        g = 0.5 * rmad
        return g

    def calculateMoranI(self, sim):
        moran_I = 0
        average_income = statistics.mean([agent.income for agent in sim.agents])
        income_variance = np.var([agent.income for agent in sim.agents])
        list_houses = []
        sum_w = 0

        for i in sim.houses:
            for j in i:
                list_houses.append(j)

        weights_matrix = [[0]*len(list_houses) for i in range(len(list_houses))]

        for i in range(len(list_houses)):
            for j in range(len(list_houses)):
                if list_houses[j].location in [pos for pos in list_houses[i].neighbourhood]:
                    weights_matrix[i][j] = 1
                    sum_w += 1
                else:
                    weights_matrix[i][j] = 0

        sum_of_differences = 0

        for i in range(len(list_houses)):
            for j in range(len(list_houses)):
                sum_of_differences += weights_matrix[i][j] * (sim.ledger[list_houses[i]].income - average_income) * (sim.ledger[list_houses[j]].income - average_income)

        moran_I = sum_of_differences / (income_variance * sum_w)

        return moran_I

if __name__ == "__main__":
    n_agents = 400    # Number of households (constant across periods). = x*y
    grid_size = [20, 20]  # Size of city (number of houses = x*y) this is so that we can accept non-square cities.

    income_range = [1000, 10000]  # Range of period incomes.
    n_periods = 10  # Number of periods of bidding (number of periods of time between movements)

    new_sim = Simulation(n_agents, income_range, grid_size, n_periods)

    root = tk.Tk()

    Display(new_sim, grid_size, root).pack(side="top", fill="both", expand=True)
    root.mainloop()

